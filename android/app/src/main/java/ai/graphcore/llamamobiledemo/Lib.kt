// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

package ai.graphcore.llamamobiledemo

import android.os.Handler
import android.os.Looper
import java.nio.ByteBuffer
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.thread
import kotlin.concurrent.withLock
import kotlin.time.DurationUnit
import kotlin.time.TimeSource

enum class Model(
    val label: String,
    val supportsImage: Boolean,
    val sizeLabel: String? = null,
    val fileName: String? = null,
    val downloadUrl: String? = null
) {
    Dummy("Select Model", true),
    VisionS3d8(
        "Vision (S3D8)",
        true,
        sizeLabel = "3.7 GB",
        fileName = "vision-11B-s3d8.sqt",
        downloadUrl = "https://graphcore-research-public.s3.eu-west-1.amazonaws.com/2026-llama-mobile/models/20260611/vision-11B-s3d8.sqt"
    ),
    TextInt8(
        "Text (INT8)",
        false,
        sizeLabel = "1.5 GB",
        fileName = "text-1B-int8.sqt",
        downloadUrl = "https://graphcore-research-public.s3.eu-west-1.amazonaws.com/2026-llama-mobile/models/20260611/text-1B-int8.sqt"
    ),
}

data class Image(
    val width: Int,
    val height: Int,
    val argb: ByteBuffer
)

object Settings {
    val maxGeneratedTokens = 256
    val temperature = 0.6
    val topK = 50
    val topP = 0.9
}

enum class ProgressPhase {
    Loading,
    ImagePrefill,
    Prefill
}

interface Generator {
    fun load(path: String)
    fun unload()
    fun progress(): Double?
    fun prefillImage(
        imageWidth: Int,
        imageHeight: Int,
        imageArgb: ByteBuffer
    )
    fun clearImagePrefill()
    fun prefillText(
        prefix: String,
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
    external override fun prefillImage(
        imageWidth: Int,
        imageHeight: Int,
        imageArgb: ByteBuffer
    )
    external override fun clearImagePrefill()
    external override fun prefillText(
        prefix: String,
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
    private var imageDescription: String? = null
    private val tokens = listOf(
        "Response: ", "I'm ", "a ", "dummy", "model", ", ",
        "I ", "have ", "no ", "wise ", "words", "."
    )

    override fun load(path: String) {
        simulateProgress(5, 20)
    }
    override fun unload() {
        imageDescription = null
    }
    override fun progress(): Double? = currentProgress

    override fun prefillImage(
        imageWidth: Int,
        imageHeight: Int,
        imageArgb: ByteBuffer
    ) {
        simulateProgress(5, 50)
        imageDescription = "${imageWidth}x$imageHeight RGB"
    }

    override fun clearImagePrefill() {
        imageDescription = null
    }

    override fun prefillText(
        prefix: String,
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
        return arrayOf(prefix, "Prompt: \"$prefix\"\nImage: ${imageDescription ?: "None"}\n")
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
    sealed interface Command {
        data class Load(val model: Model, val path: String) : Command
        data class PrefillImage(val imageId: Long, val image: Image) : Command
        object ClearImagePrefill : Command
        data class Generate(
            val prompt: String,
            val imageId: Long? = null,
            val image: Image? = null
        ) : Command
        object Stop : Command
    }

    sealed interface Event {
        data class Loaded(val model: Model) : Event
        data class ImagePrefillStarted(val imageId: Long) : Event
        data class ImagePrefillFinished(val imageId: Long) : Event
        object GenerationStarted : Event
        object GenerationFinished : Event
        data class Progress(
            val phase: ProgressPhase,
            val progress: Double?,
            val imageId: Long?
        ) : Event
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
            when (command) {
                is Command.Load -> {
                    commands.clear()
                    commands.add(command)
                }
                Command.Stop -> {
                    commands.removeIf { it !is Command.Load }
                    commands.add(command)
                }
                is Command.PrefillImage -> {
                    commands.removeIf {
                        it is Command.PrefillImage ||
                            it is Command.Generate ||
                            it is Command.ClearImagePrefill
                    }
                    commands.add(command)
                }
                Command.ClearImagePrefill -> {
                    commands.removeIf {
                        it is Command.PrefillImage ||
                            it is Command.Generate ||
                            it is Command.ClearImagePrefill
                    }
                    commands.add(command)
                }
                is Command.Generate -> {
                    commands.removeIf { it is Command.Generate || it is Command.Stop }
                    commands.add(command)
                }
            }
            hasCommand.signal()
        }
    }

    private val lock = ReentrantLock()
    private val hasCommand = lock.newCondition()
    private val mainLooper = Handler(Looper.getMainLooper())
    @Volatile
    private var commandListener: (Event) -> Unit = {}
    private val commands = ArrayDeque<Command>()
    private var generator: Generator = DummyGenerator
    private var cachedImageId: Long? = null

    private fun onEvent(event: Event) {
        mainLooper.post {
            commandListener(event)
        }
    }

    private fun shouldInterruptGeneration(): Boolean {
        return lock.withLock { commands.isNotEmpty() }
    }

    private fun prefillImage(imageId: Long, image: Image) {
        onEvent(Event.ImagePrefillStarted(imageId))
        cachedImageId = null
        try {
            withProgress(ProgressPhase.ImagePrefill, imageId) {
                generator.prefillImage(
                    imageWidth = image.width,
                    imageHeight = image.height,
                    imageArgb = image.argb
                )
            }
            cachedImageId = imageId
        } finally {
            onEvent(Event.ImagePrefillFinished(imageId))
        }
    }

    private fun generate(command: Command.Generate) {
        try {
            if (command.imageId != null && cachedImageId != command.imageId) {
                val image = command.image
                    ?: throw RuntimeException("Image prefill has not completed")
                prefillImage(command.imageId, image)
            }
            if (command.imageId == null && cachedImageId != null) {
                cachedImageId = null
                generator.clearImagePrefill()
            }
            if (shouldInterruptGeneration()) return

            onEvent(Event.GenerationStarted)
            val timer = TimeSource.Monotonic
            val tStart = timer.markNow()
            val parts = withProgress(ProgressPhase.Prefill) {
                generator.prefillText(
                    command.prompt,
                    maxGeneratedTokens = Settings.maxGeneratedTokens,
                    temperature = Settings.temperature,
                    topK = Settings.topK,
                    topP = Settings.topP
                )
            }
            if (shouldInterruptGeneration()) return

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

            for (i in 1..Settings.maxGeneratedTokens) {
                if (shouldInterruptGeneration()) return
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
        } finally {
            onEvent(Event.GenerationFinished)
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
                    cachedImageId = null
                    withProgress(ProgressPhase.Loading) {
                        generator.load(command.path)
                    }
                    onEvent(Event.Loaded(command.model))
                }

                is Command.PrefillImage -> {
                    prefillImage(command.imageId, command.image)
                }

                Command.ClearImagePrefill -> {
                    cachedImageId = null
                    generator.clearImagePrefill()
                }

                is Command.Generate -> {
                    generate(command)
                }

                Command.Stop -> {}
            }
        } catch (error: Throwable) {
            if (command is Command.Load) {
                try {
                    generator.unload()
                } catch (_: Throwable) { }
                generator = DummyGenerator
                generator.load("")
                onEvent(Event.Loaded(Model.Dummy))
            }
            onEvent(Event.Error(error.message ?: error.toString()))
        }
    }

    private fun <T> withProgress(
        phase: ProgressPhase,
        imageId: Long? = null,
        block: () -> T
    ): T {
        val running = AtomicBoolean(true)
        val poller = thread {
            var lastProgress: Double? = null
            while (running.get()) {
                val progress = generator.progress()
                if (progress != null && progress != lastProgress) {
                    lastProgress = progress
                    onEvent(Event.Progress(phase, progress, imageId))
                }
                Thread.sleep(100)
            }
        }
        try {
            return block()
        } finally {
            running.set(false)
            poller.join()
            onEvent(Event.Progress(phase, null, imageId))
        }
    }

    init {
        thread {
            while (true) {
                // Wait for the next command
                val command = lock.withLock {
                    while (commands.isEmpty()) {
                        hasCommand.await()
                    }
                    commands.removeFirst()
                }
                // Don't do this while holding the lock!
                handleCommand(command)
            }
        }
    }
}
