package com.echomebetter.engine.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.echomebetter.engine.generation.Seq2SeqRunner
import com.echomebetter.engine.generation.StepResult
import com.echomebetter.engine.manifest.Architecture
import java.nio.FloatBuffer
import java.nio.LongBuffer

class ContractException(message: String) : Exception("model does not follow the cross-kv/1 contract: $message")

/**
 * The cached cross-attention I/O contract (`ioContract` in model.json) on top of onnxruntime sessions:
 *
 *     encoder.onnx  input_ids, attention_mask  →  cross_key.{i}, cross_value.{i}
 *     decoder.onnx  input_ids, encoder_attention_mask, past_key.{i}, past_value.{i}, cross_key.{i}, cross_value.{i}
 *                   →  logits, present_key.{i}, present_value.{i}
 *
 * The encoder produces every decoder layer's cross-attention keys/values once; each decoder step then processes only
 * the newest token, feeding the previous step's `present_*` back in as `past_*`. Tensor names are checked against the
 * sessions on construction, so a model exported for a different contract fails at once rather than mid-generation.
 *
 * A style's adapter is a pair of LoRA weight sets, one per graph, made active through the run options: the encoder
 * half on the encoder run, the decoder half on every decoder step. Without them the runner is the plain base model.
 *
 * Not thread-safe: one generation at a time.
 */
class CrossKvRunner(
    private val env: OrtEnvironment,
    private val encoder: OrtSession,
    private val decoder: OrtSession,
    private val geometry: Architecture,
    private val adapter: LoraAdapterPair? = null,
) : Seq2SeqRunner<CrossKvRunner.EncoderMemory, CrossKvRunner.DecoderCache> {

    /** The encoder's output for one prompt; owns its tensors until released. */
    class EncoderMemory internal constructor(internal val mask: OnnxTensor, internal val result: OrtSession.Result, internal val cross: Map<String, OnnxValue>)

    /** The past keys/values for the next step: the empty initial tensors, or the outputs of the step before. */
    class DecoderCache internal constructor(internal val past: Map<String, OnnxValue>, internal val owner: AutoCloseable)

    private val crossNames = (0 until geometry.numDecoderLayers).flatMap { listOf("cross_key.$it", "cross_value.$it") }
    private val pastNames = (0 until geometry.numDecoderLayers).flatMap { listOf("past_key.$it" to "present_key.$it", "past_value.$it" to "present_value.$it") }

    init {
        requireNames(encoder.inputNames, listOf("input_ids", "attention_mask"), "encoder inputs")
        requireNames(encoder.outputNames, crossNames, "encoder outputs")
        requireNames(decoder.inputNames, listOf("input_ids", "encoder_attention_mask") + pastNames.map { it.first } + crossNames, "decoder inputs")
        requireNames(decoder.outputNames, listOf("logits") + pastNames.map { it.second }, "decoder outputs")
    }

    /** The same sessions, with [adapter] active on every run. */
    fun withAdapter(adapter: LoraAdapterPair): CrossKvRunner = CrossKvRunner(env, encoder, decoder, geometry, adapter)

    private fun ids(values: List<Int>): OnnxTensor =
        OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(values.size) { values[it].toLong() }), longArrayOf(1, values.size.toLong()))

    override fun encode(inputIds: List<Int>): EncoderMemory {
        val mask = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(inputIds.size) { 1L }), longArrayOf(1, inputIds.size.toLong()))
        try {
            ids(inputIds).use { input ->
                val result = encoder.run(mapOf("input_ids" to input, "attention_mask" to mask), adapter?.encoderOptions ?: NO_OPTIONS)
                return EncoderMemory(mask, result, crossNames.associateWith { result.get(it).get() })
            }
        } catch (error: Throwable) {
            mask.close()
            throw error
        }
    }

    override fun initialCache(): DecoderCache {
        val shape = longArrayOf(1, geometry.numHeads.toLong(), 0, geometry.headDim.toLong())
        val tensors = pastNames.associate { (past, _) -> past to OnnxTensor.createTensor(env, FloatBuffer.allocate(0), shape) }
        return DecoderCache(tensors) { tensors.values.forEach(OnnxTensor::close) }
    }

    override fun step(tokenId: Int, memory: EncoderMemory, cache: DecoderCache): StepResult<DecoderCache> {
        val result = ids(listOf(tokenId)).use { input ->
            val inputs = HashMap<String, OnnxTensor>(2 + cache.past.size + memory.cross.size)
            inputs["input_ids"] = input
            inputs["encoder_attention_mask"] = memory.mask
            cache.past.forEach { (name, value) -> inputs[name] = value as OnnxTensor }
            memory.cross.forEach { (name, value) -> inputs[name] = value as OnnxTensor }
            decoder.run(inputs, adapter?.decoderOptions ?: NO_OPTIONS)
        }
        try {
            val logits = result.get("logits").get() as OnnxTensor
            val shape = logits.info.shape
            val vocab = shape.last().toInt()
            val data = logits.floatBuffer
            // Copy the last position's scores: the processors change them in place.
            val scores = FloatArray(vocab)
            data.position(data.limit() - vocab)
            data.get(scores)
            val next = DecoderCache(pastNames.associate { (past, present) -> past to result.get(present).get() }, result)
            // The step's inputs are spent: the previous step's outputs (or the empty initial tensors) can go.
            cache.owner.close()
            return StepResult(scores, next)
        } catch (error: Throwable) {
            result.close()
            throw error
        }
    }

    override fun release(memory: EncoderMemory, cache: DecoderCache) {
        cache.owner.close()
        memory.result.close()
        memory.mask.close()
    }

    private companion object {
        val NO_OPTIONS = OrtSession.RunOptions()

        fun requireNames(actual: Set<String>, expected: List<String>, where: String) {
            val missing = expected.filterNot { it in actual }
            if (missing.isNotEmpty()) throw ContractException("$where is missing ${missing.take(4).joinToString()}")
        }
    }
}
