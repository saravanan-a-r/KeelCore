package com.echomebetter.engine.prompt

import com.echomebetter.engine.manifest.TextTransform

/**
 * Reversible text transforms a model can ask for in its manifest (`inputTransforms`): applied to the user's text
 * before tokenizing, undone on the model's output, last-applied first.
 */
internal object TextTransforms {
    /**
     * Thelivu's SentencePiece decoder turns every "▁" into a space, so a literal "▁" in the user's text would not
     * survive a round trip. The text is escaped into Private Use Area code points first, exactly as the training
     * pipeline does (tokenizer/training/marker_escape.py): one left-to-right pass, escaping the escape character
     * too, so the scheme is lossless for any input.
     */
    private const val SPIECE_UNDERLINE = '▁'
    private const val ESC = ''
    private const val ESCAPED_UNDERLINE = ''

    fun escapeSpieceMarkers(text: String): String {
        if (ESC !in text && SPIECE_UNDERLINE !in text) return text
        val out = StringBuilder(text.length + 8)
        for (char in text) {
            when (char) {
                ESC -> out.append(ESC).append(ESC)
                SPIECE_UNDERLINE -> out.append(ESC).append(ESCAPED_UNDERLINE)
                else -> out.append(char)
            }
        }
        return out.toString()
    }

    fun unescapeSpieceMarkers(text: String): String {
        if (ESC !in text) return text
        val out = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val char = text[i]
            val next = text.getOrNull(i + 1)
            if (char == ESC && (next == ESC || next == ESCAPED_UNDERLINE)) {
                out.append(if (next == ESC) ESC else SPIECE_UNDERLINE)
                i += 2
            } else {
                out.append(char)
                i++
            }
        }
        return out.toString()
    }

    fun apply(text: String, transforms: List<TextTransform>): String =
        transforms.fold(text) { current, transform ->
            when (transform) {
                TextTransform.ESCAPE_SPIECE_MARKERS -> escapeSpieceMarkers(current)
            }
        }

    fun undo(text: String, transforms: List<TextTransform>): String =
        transforms.asReversed().fold(text) { current, transform ->
            when (transform) {
                TextTransform.ESCAPE_SPIECE_MARKERS -> unescapeSpieceMarkers(current)
            }
        }
}
