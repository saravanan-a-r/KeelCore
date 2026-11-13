package com.echomebetter.engine.generation

import com.echomebetter.engine.manifest.GenerationSettings
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

/** Scores over the vocabulary for the next token, and the cache for the step after. */
class StepResult<Cache>(val logits: FloatArray, val cache: Cache)

/** An encoder-decoder that encodes a prompt once and then advances one token at a time. */
interface Seq2SeqRunner<Memory, Cache> {
    fun encode(inputIds: List<Int>): Memory

    fun initialCache(): Cache

    fun step(tokenId: Int, memory: Memory, cache: Cache): StepResult<Cache>

    /** Release what [memory] and [cache] hold; called once generation ends, however it ends. */
    fun release(memory: Memory, cache: Cache)
}

class GreedyOptions(
    val settings: GenerationSettings,
    val decoderStartId: Int,
    val eosId: Int,
    val maxNewTokens: Int,
    /** Token id → vocabulary piece; lets the decoder recognise a URL while it is written. */
    val piece: (Int) -> String?,
    /** After each generated token (EOS excluded), with every token generated so far (the decoder's own list: read only). */
    val onToken: ((count: Int, generated: List<Int>) -> Unit)? = null,
)

class GreedyResult(
    /** Generated IDs, EOS excluded. */
    val ids: List<Int>,
    /** True when EOS was produced; false when maxNewTokens cut generation off. */
    val finished: Boolean,
)

/**
 * Greedy decoding with a KV cache, independent of any inference runtime. Cancellation is checked before every step,
 * so cancelling the calling coroutine stops generation within one decoder step.
 */
object GreedyDecoder {
    suspend fun <Memory, Cache> generate(runner: Seq2SeqRunner<Memory, Cache>, inputIds: List<Int>, options: GreedyOptions): GreedyResult {
        coroutineContext.ensureActive()
        val memory = runner.encode(inputIds)
        var cache = runner.initialCache()
        val sequence = arrayListOf(options.decoderStartId)
        val generated = ArrayList<Int>()
        val allowance = NgramAllowance.of(options.settings, options.piece, inputIds)
        try {
            repeat(options.maxNewTokens) {
                coroutineContext.ensureActive()
                val result = runner.step(sequence.last(), memory, cache)
                cache = result.cache
                val next = LogitsProcessors.argmax(LogitsProcessors.process(result.logits, sequence, options.settings, allowance))
                sequence.add(next)
                if (next == options.eosId) return GreedyResult(generated, finished = true)
                generated.add(next)
                allowance.urls?.observe(next)
                options.onToken?.invoke(generated.size, generated)
            }
            return GreedyResult(generated, finished = false)
        } finally {
            runner.release(memory, cache)
        }
    }
}
