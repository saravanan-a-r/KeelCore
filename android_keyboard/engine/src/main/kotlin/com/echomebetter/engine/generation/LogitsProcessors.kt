package com.echomebetter.engine.generation

import com.echomebetter.engine.manifest.GenerationSettings

/**
 * HuggingFace's greedy-search logits processors, for a single hypothesis, applied in HuggingFace's order
 * (repetition penalty, then no-repeat n-gram). A port of chrome_extension/src/engine/generation/logitsProcessors.ts.
 *
 * `sequence` is the decoder input so far, decoder-start token included (the same `input_ids` HuggingFace passes its
 * processors), so a manifest's generation settings mean exactly what they mean in `model.generate`.
 */
internal object LogitsProcessors {
    /**
     * A word the model is writing is a URL once it begins like one: `scheme://`, `www.`, or `host.tld/`, or is about
     * to: the scheme or `www` written so far (`https`, `https:`, `www`). SFT_adapter_rephrase/src/decoding.py and the
     * extension hold the same pattern; keep the three identical.
     */
    private val URL_WORD = Regex("""^(?:[a-z][a-z0-9+.\-]*://|www\.|(?:[a-z0-9\-]+\.)+[a-z]{2,}/|(?:https?|ftp)(?::/?)?$|www$)""", RegexOption.IGNORE_CASE)
    private val BYTE_PIECE = Regex("^<0x([0-9A-F]{2})>$")

    /** A vocabulary piece as plain text: `▁` is a space, a byte piece is its ASCII character (else U+FFFD). */
    fun pieceText(piece: String): String {
        val byte = BYTE_PIECE.matchEntire(piece)
        if (byte != null) {
            val value = byte.groupValues[1].toInt(16)
            return if (value < 0x80) value.toChar().toString() else "�"
        }
        return piece.replace('▁', ' ')
    }

    fun isUrlWord(word: String): Boolean = URL_WORD.containsMatchIn(word)

    fun applyRepetitionPenalty(scores: FloatArray, sequence: List<Int>, penalty: Double) {
        if (penalty == 1.0) return
        // In double, then rounded to float: what the extension's Float32Array store does, so ties break the same way.
        for (token in sequence.toSet()) {
            val score = scores[token].toDouble()
            scores[token] = (if (score < 0) score * penalty else score / penalty).toFloat()
        }
    }

    fun bannedNgramTokens(sequence: List<Int>, size: Int, allowance: NgramAllowance? = null): List<Int> {
        if (size <= 0 || sequence.size + 1 < size) return emptyList()
        val prefixStart = sequence.size + 1 - size
        val seen = LinkedHashMap<Int, Int>() // token that completed the prefix before -> times
        for (start in 0..sequence.size - size) {
            var matches = true
            for (k in 0 until size - 1) {
                if (sequence[start + k] != sequence[prefixStart + k]) {
                    matches = false
                    break
                }
            }
            if (!matches) continue
            val token = sequence[start + size - 1]
            if (allowance != null && token in allowance.exempt) continue
            seen[token] = (seen[token] ?: 0) + 1
        }
        // Inside a URL an n-gram may repeat, but only as often as the input has it.
        val inUrl = allowance?.urls?.inside == true
        val prefix = sequence.subList(prefixStart, sequence.size)
        return seen.filter { (token, times) -> !(inUrl && times < (allowance?.inputNgrams?.get(prefix + token) ?: 0)) }.keys.toList()
    }

    fun process(scores: FloatArray, sequence: List<Int>, settings: GenerationSettings, allowance: NgramAllowance?): FloatArray {
        applyRepetitionPenalty(scores, sequence, settings.repetitionPenalty)
        for (token in bannedNgramTokens(sequence, settings.noRepeatNgramSize, allowance)) scores[token] = Float.NEGATIVE_INFINITY
        return scores
    }

    /** Index of the largest score; the first one on ties, like numpy/torch argmax. */
    fun argmax(scores: FloatArray): Int {
        var best = 0
        for (i in 1 until scores.size) if (scores[i] > scores[best]) best = i
        return best
    }
}

/**
 * Whether the word being written is a URL, tracked token by token.
 *
 * A URL repeats its own pieces on purpose (the host in two links, `/docs/` in a path, `utm_` in each parameter), so
 * the n-gram ban corrupts it (`utm_medium` came out `utMid`). While [inside] is true the ban is lifted; the first
 * space ends the word and restores it for the text around the URL.
 */
internal class UrlTracker(private val textOf: (Int) -> String) {
    private var word = ""

    fun observe(token: Int) {
        val text = textOf(token)
        val space = LAST_WORD.find(text)?.range?.first ?: -1
        word = if (space >= 0) text.substring(space + 1) else word + text
    }

    val inside: Boolean get() = LogitsProcessors.isUrlWord(word)

    private companion object {
        /** The last whitespace character and the word after it; JavaScript's `\s` is Unicode-aware, so this one is too. */
        val LAST_WORD = Regex("""(?U)\s\S*$""")
    }
}

/** What the n-gram ban lets through: built once per request, read on every step. */
internal class NgramAllowance(
    val exempt: Set<Int>,
    /** Present when URLs are exempt; the decoder feeds it every generated token. */
    val urls: UrlTracker? = null,
    /** With [urls]: how often each n-gram occurs in the input. */
    val inputNgrams: Map<List<Int>, Int>? = null,
) {
    companion object {
        /** [piece] maps a token id to its vocabulary piece and [inputIds] is the encoder input. */
        fun of(settings: GenerationSettings, piece: (Int) -> String?, inputIds: List<Int>): NgramAllowance {
            if (!settings.noRepeatExemptUrls) return NgramAllowance(settings.noRepeatExemptTokenIds)
            val size = maxOf(settings.noRepeatNgramSize, 1)
            val counts = HashMap<List<Int>, Int>()
            for (start in 0..inputIds.size - size) {
                val key = inputIds.subList(start, start + size).toList()
                counts[key] = (counts[key] ?: 0) + 1
            }
            return NgramAllowance(
                exempt = settings.noRepeatExemptTokenIds,
                urls = UrlTracker { token -> LogitsProcessors.pieceText(piece(token) ?: "") },
                inputNgrams = counts,
            )
        }
    }
}
