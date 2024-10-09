package ai.graphcore.squashedllama

import android.os.Handler
import android.os.Looper
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.thread
import kotlin.concurrent.withLock
import kotlin.time.DurationUnit
import kotlin.time.TimeSource

object Lib {
    external fun load(path: String)
    external fun unload()
    external fun prefill(prefix: String, maxGeneratedTokens: Int): Array<String>
    external fun generate(): String

    init {
        System.loadLibrary("squashed-llama")
    }
}

object Worker {
    val maxGeneratedTokens = 8

    interface Command {
        data class Load(val path: String) : Command
        data object Unload : Command
        data class Generate(val prompt: String) : Command
    }

    interface Event {
        data class Loaded(val path: String?) : Event
        data class Response(
            val text: String,
            val prompt: String,
            val prefillRate: Double,
            val generationRate: Double?
        ) : Event
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

    private fun onEvent(event: Event) {
        mainLooper.post {
            commandListener(event)
        }
    }

    private fun handleCommand(command: Command) {
        when (command) {
            is Command.Load -> {
                Lib.load(command.path)
                onEvent(Event.Loaded(command.path))
            }

            is Command.Unload -> {
                Lib.unload()
                onEvent(Event.Loaded(null))
            }

            is Command.Generate -> {
                // Prefill
                val timer = TimeSource.Monotonic
                val tStart = timer.markNow()
                val parts = Lib.prefill(command.prompt, maxGeneratedTokens)
                var response = parts.last()
                val tPrefill = timer.markNow()
                val prefillRate =
                    parts.size.toDouble() / (tPrefill - tStart).toDouble(DurationUnit.SECONDS)
                onEvent(
                    Event.Response(
                        text = response,
                        prompt = command.prompt,
                        prefillRate = prefillRate,
                        generationRate = null
                    )
                )

                // Generation
                for (i in 1..maxGeneratedTokens) {
                    if (nextCommand != null) return;  // Interrupt generation
                    response += Lib.generate()
                    val tGenerate = timer.markNow()
                    val generationRate =
                        i / (tGenerate - tPrefill).toDouble(DurationUnit.SECONDS)
                    onEvent(
                        Event.Response(
                            text = response,
                            prompt = command.prompt,
                            prefillRate = prefillRate,
                            generationRate = generationRate
                        )
                    )
                }
            }
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
