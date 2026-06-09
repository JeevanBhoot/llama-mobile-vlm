// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

package ai.graphcore.squashedllama

import android.os.Handler
import android.os.Looper
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.thread
import kotlin.concurrent.withLock
import kotlin.time.DurationUnit
import kotlin.time.TimeSource

enum class Model(val label: String, val path: String, val supportsImage: Boolean) {
    Dummy("None", "", true),
    TextInt8("Text (INT8)", "/data/local/tmp/text-int8.sqt", false),
    VisionS3d8("Vision (S3D8)", "/data/local/tmp/vision-s3d8-proud-sponge-1878.sqt", true),
}

data class Image(
    val width: Int,
    val height: Int,
    val argb: ByteBuffer
)

object Settings {
    val maxGeneratedTokens = 100
    val temperature = 0.0  // Greedy for testing
    val topK = 50
    val topP = 1.0
}

enum class ProgressPhase {
    Loading,
    Prefill
}

interface Generator {
    fun load(path: String)
    fun unload()
    fun progress(): Double?
    fun prefill(
        prefix: String,
        imageWidth: Int,
        imageHeight: Int,
        imageArgb: ByteBuffer?,
        maxGeneratedTokens: Int,
        temperature: Double,
        topK: Int,
        topP: Double
    ): Array<String>

    fun generate(): String
}

object Lib : Generator {
    external override fun load(path: String)
    external override fun unload()
    external override fun progress(): Double?
    external override fun prefill(
        prefix: String,
        imageWidth: Int,
        imageHeight: Int,
        imageArgb: ByteBuffer?,
        maxGeneratedTokens: Int,
        temperature: Double,
        topK: Int,
        topP: Double
    ): Array<String>
    external override fun generate(): String

    init {
        System.loadLibrary("squashed-llama")
    }
}

object DummyGenerator : Generator {
    private var nextToken = 0
    @Volatile
    private var currentProgress: Double? = null
    private val tokens = listOf(
        "Response: ", "I'm ", "a ", "dummy", ", ", "I ", "have ", "no ", "wise ", "words", "."
    )

    override fun load(path: String) {
        simulateProgress(5, 20)
    }
    override fun unload() { }
    override fun progress(): Double? = currentProgress

    override fun prefill(
        prefix: String,
        imageWidth: Int,
        imageHeight: Int,
        imageArgb: ByteBuffer?,
        maxGeneratedTokens: Int,
        temperature: Double,
        topK: Int,
        topP: Double
    ): Array<String> {
        if (prefix.trim().lowercase() == "error") {
            throw RuntimeException("You asked me for an error - you've got it!")
        }
        simulateProgress(5, 50)
        nextToken = 0
        val hasImage = imageArgb != null && imageWidth > 0 && imageHeight > 0
        val imageDescription = if (hasImage) {
            "${imageWidth}x$imageHeight RGB"
        } else {
            "None"
        }
        return arrayOf(prefix, "Prompt: \"$prefix\"\nImage: $imageDescription\n")
    }
    override fun generate(): String {
        Thread.sleep(100)
        return tokens.getOrElse(nextToken++) { "" }
    }

    private fun simulateProgress(steps: Int, stepMs: Long) {
        currentProgress = 0.0
        try {
            for (i in 1..steps) {
                Thread.sleep(stepMs)
                currentProgress = i.toDouble() / steps
            }
        } finally {
            currentProgress = null
        }
    }
}

object Worker {
    interface Command {
        data class Load(val model: Model) : Command
        data class Generate(
            val prompt: String,
            val image: Image? = null
        ) : Command
    }

    interface Event {
        data class Loaded(val model: Model) : Event
        data class Progress(val phase: ProgressPhase, val progress: Double?) : Event
        data class Response(
            val text: String,
            val prompt: String,
            val prefillTime: Double,
            val generationRate: Double?
        ) : Event
        data class Error(val message: String) : Event
    }

    fun setListener(listener: (Event) -> Unit) {
        commandListener = listener
    }

    fun send(command: Command) {
        lock.withLock {
            // Always preempt, but generate must not preempt Load
            if (!(command is Command.Generate && nextCommand is Command.Load)) {
                nextCommand = command
                hasCommand.signal()
            }
        }
    }

    private val lock = ReentrantLock()
    private val hasCommand = lock.newCondition()
    private val mainLooper = Handler(Looper.getMainLooper())
    @Volatile
    private var nextCommand: Command? = null
    @Volatile
    private var commandListener: (Event) -> Unit = {}
    private var generator: Generator = DummyGenerator

    private fun onEvent(event: Event) {
        mainLooper.post {
            commandListener(event)
        }
    }

    private fun handleCommand(command: Command) {
        try {
            when (command) {
                is Command.Load -> {
                    val nextGenerator = if (command.model == Model.Dummy) DummyGenerator else Lib
                    if (generator !== nextGenerator) {
                        generator.unload()
                        generator = nextGenerator
                    }
                    withProgress(ProgressPhase.Loading) {
                        generator.load(command.model.path)
                    }
                    onEvent(Event.Loaded(command.model))
                }

                is Command.Generate -> {
                    // Prefill
                    val timer = TimeSource.Monotonic
                    val tStart = timer.markNow()
                    val parts = withProgress(ProgressPhase.Prefill) {
                        generator.prefill(
                            command.prompt,
                            imageWidth = command.image?.width ?: 0,
                            imageHeight = command.image?.height ?: 0,
                            imageArgb = command.image?.argb,
                            maxGeneratedTokens = Settings.maxGeneratedTokens,
                            temperature = Settings.temperature,
                            topK = Settings.topK,
                            topP = Settings.topP
                        )
                    }
                    var response = parts.last()
                    val tPrefill = timer.markNow()
                    val prefillTime = (tPrefill - tStart).toDouble(DurationUnit.SECONDS)
                    onEvent(
                        Event.Response(
                            text = response,
                            prompt = command.prompt,
                            prefillTime = prefillTime,
                            generationRate = null
                        )
                    )

                    // Generation
                    for (i in 1..Settings.maxGeneratedTokens) {
                        if (nextCommand != null) return;  // Interrupt generation
                        val token = generator.generate()
                        if (token.isEmpty()) break
                        response += token
                        val tGenerate = timer.markNow()
                        val generationRate =
                            i / (tGenerate - tPrefill).toDouble(DurationUnit.SECONDS)
                        onEvent(
                            Event.Response(
                                text = response,
                                prompt = command.prompt,
                                prefillTime = prefillTime,
                                generationRate = generationRate
                            )
                        )
                    }
                }
            }
        } catch (error: Throwable) {
            if (command is Command.Load) {
                try {
                    generator.unload()
                } catch (_: Throwable) { }
                generator = DummyGenerator
                generator.load(Model.Dummy.path)
                onEvent(Event.Loaded(Model.Dummy))
            }
            onEvent(Event.Error(error.message ?: error.toString()))
        }
    }

    private fun <T> withProgress(phase: ProgressPhase, block: () -> T): T {
        val running = AtomicBoolean(true)
        val poller = thread {
            var lastProgress: Double? = null
            while (running.get()) {
                val progress = generator.progress()
                if (progress != null && progress != lastProgress) {
                    lastProgress = progress
                    onEvent(Event.Progress(phase, progress))
                }
                Thread.sleep(100)
            }
        }
        try {
            return block()
        } finally {
            running.set(false)
            poller.join()
            onEvent(Event.Progress(phase, null))
        }
    }

    init {
        thread {
            while (true) {
                // Wait for the next command
                val command = lock.withLock {
                    while (nextCommand == null) {
                        hasCommand.await()
                    }
                    val command = nextCommand!!
                    nextCommand = null
                    command
                }
                // Don't do this while holding the lock!
                handleCommand(command)
            }
        }
    }
}
