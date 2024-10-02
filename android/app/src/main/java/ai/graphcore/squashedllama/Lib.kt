package ai.graphcore.squashedllama

object Lib {
    external fun meaning(): Int

    init {
        System.loadLibrary("squashed-llama")
    }
}
