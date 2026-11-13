package com.echomebetter.engine.prompt

import com.echomebetter.engine.RewriteException
import com.echomebetter.engine.manifest.BaseModel
import com.echomebetter.engine.manifest.PromptSegment
import com.echomebetter.engine.tokenizer.Tokenizer

/**
 * Turns (style token, user text) into the encoder's input IDs, following the manifest's prompt frame. Thelivu's
 * frame is `<style:X> <text_to_rewrite> text </text_to_rewrite> </s>`, exactly as the adapters were trained
 * (SFT_adapter_rephrase/src/codec.py).
 *
 * Control pieces are ordinary vocabulary entries, so user text that spells one would encode to it and break the
 * frame; such text is refused, as the training data was, instead of handing the model a marker mid-sentence.
 */
internal class PromptBuilder(private val base: BaseModel, private val tokenizer: Tokenizer) {
    private val frameIds: Map<String, Int>
    private val reservedIds: Set<Int>

    init {
        val named = base.prompt.filterIsInstance<PromptSegment.Token>().map { it.token } + base.reservedInputTokens
        frameIds = named.associateWith { idOf(it) }
        reservedIds = base.reservedInputTokens.map { frameIds.getValue(it) }.toSet()
    }

    /** The token's id, failing when the vocabulary does not have it (a model.json for another tokenizer). */
    fun idOf(token: String): Int =
        tokenizer.tokenToId(token) ?: throw IllegalStateException("model.json: token \"$token\" is not in the tokenizer's vocabulary")

    /** The encoder input for [text], rewritten under [styleToken]. */
    fun build(styleToken: String, text: String): List<Int> {
        val styleId = idOf(styleToken)
        val ids = ArrayList<Int>()
        for (segment in base.prompt) {
            when (segment) {
                is PromptSegment.Token -> ids.add(frameIds.getValue(segment.token))
                PromptSegment.StyleToken -> ids.add(styleId)
                PromptSegment.Input -> {
                    val inputIds = encodeInput(text)
                    if (inputIds.any { it in reservedIds }) throw RewriteException.ReservedMarkup()
                    ids.addAll(inputIds)
                }
            }
        }
        val limit = base.limits.maxInputTokens
        if (ids.size > limit) throw RewriteException.InputTooLong(ids.size, limit)
        return ids
    }

    /** The tokens [text] alone adds to the prompt, exactly as the frame encodes it. */
    fun encodeInput(text: String): List<Int> = tokenizer.encode(TextTransforms.apply(text, base.inputTransforms))
}
