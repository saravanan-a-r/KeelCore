package com.echomebetter.engine.tokenizer

/**
 * SentencePiece Unigram segmentation: the highest-scoring way to cover a text with vocabulary pieces (Viterbi over
 * the lattice of matches).
 *
 * A port of HuggingFace `tokenizers`' `Unigram::encode_optimized` and `Unigram::tokenize` by way of the extension
 * (chrome_extension/src/engine/tokenizer/unigram.ts), including the parts that decide *which* of several
 * equal-scoring segmentations wins (strict `>` while scanning left to right, prefixes visited shortest first), the
 * unknown-character penalty, fusing runs of unknown characters, and byte fallback. IDs must agree with the Python
 * reference exactly, so tie-breaking is part of the contract.
 */
internal class UnigramModel(
    private val pieces: List<Pair<String, Double>>,
    val unkId: Int?,
    private val byteFallback: Boolean,
) {
    private class TrieNode {
        val children = HashMap<Int, TrieNode>()

        /** Set when the path from the root to this node spells a vocabulary piece. */
        var pieceId = -1
    }

    private val pieceToId = HashMap<String, Int>(pieces.size * 2)
    private val root = TrieNode()
    private val unkScore: Double

    init {
        var minScore = Double.POSITIVE_INFINITY
        pieces.forEachIndexed { id, (piece, score) ->
            pieceToId[piece] = id
            insert(piece, id)
            if (score < minScore) minScore = score
        }
        unkScore = minScore - UNK_PENALTY
    }

    fun tokenToId(piece: String): Int? = pieceToId[piece]

    fun idToToken(id: Int): String? = pieces.getOrNull(id)?.first

    private fun insert(piece: String, id: Int) {
        var node = root
        var index = 0
        while (index < piece.length) {
            val codePoint = piece.codePointAt(index)
            node = node.children.getOrPut(codePoint) { TrieNode() }
            index += Character.charCount(codePoint)
        }
        // On duplicates the extension's map keeps the last id for the piece string; the trie keeps the same.
        node.pieceId = pieceToId.getValue(piece)
    }

    /** The best segmentation of a text, as piece strings (unknown runs fused). */
    fun segment(text: String): List<String> {
        if (text.isEmpty()) return emptyList()
        val startsAt = IntArray(text.length + 1) { -1 }
        val ids = IntArray(text.length + 1)
        val scores = DoubleArray(text.length + 1)
        startsAt[0] = 0

        var position = 0
        while (position < text.length) {
            val charLength = Character.charCount(text.codePointAt(position))
            // Like the reference, every code point boundary is processed: with an <unk> token every boundary is
            // reachable, so there is nothing to skip.
            val hereScore = scores[position]
            var hasSingleCharPiece = false
            var node: TrieNode? = root
            var index = position
            while (index < text.length) {
                val codePoint = text.codePointAt(index)
                node = node!!.children[codePoint] ?: break
                index += Character.charCount(codePoint)
                if (node.pieceId >= 0) {
                    val candidate = pieces[node.pieceId].second + hereScore
                    if (startsAt[index] == -1 || candidate > scores[index]) {
                        startsAt[index] = position
                        ids[index] = node.pieceId
                        scores[index] = candidate
                    }
                    if (index - position == charLength) hasSingleCharPiece = true
                }
            }
            if (!hasSingleCharPiece && unkId != null) {
                val end = position + charLength
                val candidate = unkScore + hereScore
                if (startsAt[end] == -1 || candidate > scores[end]) {
                    startsAt[end] = position
                    ids[end] = unkId
                    scores[end] = candidate
                }
            }
            position += charLength
        }

        val results = ArrayList<String>()
        val unknownRun = ArrayList<String>()
        var end = text.length
        while (end > 0) {
            val start = startsAt[end]
            check(start != -1) { "unigram lattice has no path (no <unk> token to fall back on)" }
            val piece = text.substring(start, end)
            if (unkId != null && ids[end] == unkId) {
                unknownRun.add(piece)
            } else {
                if (unknownRun.isNotEmpty()) {
                    results.add(unknownRun.asReversed().joinToString(""))
                    unknownRun.clear()
                }
                results.add(piece)
            }
            end = start
        }
        if (unknownRun.isNotEmpty()) results.add(unknownRun.asReversed().joinToString(""))
        results.reverse()
        return results
    }

    /** A text → token IDs. */
    fun tokenize(text: String): List<Int> {
        val out = ArrayList<Int>()
        for (piece in segment(text)) {
            val id = pieceToId[piece]
            if (id != null) {
                out.add(id)
                continue
            }
            if (byteFallback) {
                val byteIds = byteFallbackIds(piece)
                if (byteIds != null) {
                    out.addAll(byteIds)
                    continue
                }
            }
            out.add(unkId ?: error("no <unk> token for \"$piece\""))
        }
        return out
    }

    private fun byteFallbackIds(piece: String): List<Int>? =
        piece.toByteArray(Charsets.UTF_8).map { byte ->
            pieceToId["<0x%02X>".format(byte.toInt() and 0xFF)] ?: return null
        }

    private companion object {
        const val UNK_PENALTY = 10.0
    }
}
