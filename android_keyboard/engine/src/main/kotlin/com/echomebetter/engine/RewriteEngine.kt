package com.echomebetter.engine

import com.echomebetter.engine.generation.GreedyDecoder
import com.echomebetter.engine.generation.GreedyOptions
import com.echomebetter.engine.generation.GreedyResult
import com.echomebetter.engine.generation.Seq2SeqRunner
import com.echomebetter.engine.manifest.BaseModel
import com.echomebetter.engine.prompt.PromptBuilder
import com.echomebetter.engine.prompt.TextTransforms
import com.echomebetter.engine.tokenizer.Tokenizer
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** What a rewrite reports while it runs. Both calls come on the thread that runs the rewrite. */
interface RewriteListener {
    /**
     * After each generated token, with a way to read the rewrite so far. Reading decodes, so a caller showing it
     * reads only as often as it shows it. The text so far is not the result: [RewriteEngine.rewrite] returns that.
     */
    fun onToken(count: Int, readText: () -> String) {}

    /** The text so far was dropped and the rewrite starts over in pieces: a single pass ran past maxNewTokens. */
    fun onRestart() {}
}

class RewriteResult(val text: String, val inputTokens: Int, val outputTokens: Int)

/**
 * The whole rewrite path for a loaded base model: prompt → encoder → greedy decoding → text, with a style's adapter
 * active. A port of chrome_extension/src/engine/rewriteEngine.ts, runtime-agnostic in the same way: it is handed a
 * [Seq2SeqRunner] per adapter, so the tests drive it with onnxruntime on fixtures and the app with the real model.
 *
 * Not thread-safe: one rewrite at a time.
 */
class RewriteEngine(private val base: BaseModel, private val tokenizer: Tokenizer) {
    private val prompts = PromptBuilder(base, tokenizer)

    /** A run of text that goes through the model, or a separator carried through untouched. */
    private sealed interface Segment {
        val text: String

        /** [halved]: half of a chunk whose rewrite ran past maxNewTokens; it is not cut again. */
        data class Chunk(override val text: String, val halved: Boolean = false) : Segment

        data class Join(override val text: String) : Segment
    }

    /** Encoder input IDs for a request; exposed so tests can check them against the reference. */
    fun promptIds(styleToken: String, text: String): List<Int> = prompts.build(styleToken, text)

    /** Greedy generation with [runner] (the base with an adapter active, or the plain base). */
    suspend fun <M, C> generateIds(runner: Seq2SeqRunner<M, C>, inputIds: List<Int>, listener: RewriteListener? = null): GreedyResult =
        GreedyDecoder.generate(
            runner,
            inputIds,
            GreedyOptions(
                settings = base.generation,
                decoderStartId = base.tokens.decoderStartId,
                eosId = base.tokens.eosId,
                maxNewTokens = base.limits.maxNewTokens,
                piece = tokenizer::piece,
                onToken = listener?.let { l -> { count, generated -> l.onToken(count) { partialText(generated) } } },
            ),
        )

    /** Generated IDs → text, as the training codec decodes: padding skipped, input transforms undone. */
    fun decode(ids: List<Int>): String =
        TextTransforms.undo(tokenizer.decode(ids.filter { it != base.tokens.padId }), base.inputTransforms)

    /**
     * The rewrite so far, for showing while it is written. A character spelled in several byte tokens (an emoji,
     * many accented letters) is incomplete until its last byte arrives and decodes as U+FFFD meanwhile: those are
     * left off the end until it is whole.
     */
    private fun partialText(ids: List<Int>): String = decode(ids).trimEnd('�').trim()

    /**
     * One pass when the whole rewrite fits `maxNewTokens` at the style's expansion ratio, otherwise pieces from the
     * start, so no pass is spent on an output that cannot finish. A pass that still runs past the limit (an output
     * longer than the ratio allows) is redone once, in two halves.
     */
    suspend fun <M, C> rewrite(style: Style, styleToken: String, runner: Seq2SeqRunner<M, C>, text: String, listener: RewriteListener? = null): RewriteResult {
        val budget = chunkBudget(style, styleToken)
        val tokens = measureTokens(text)
        if (tokens > budget) return rewriteChunked(styleToken, runner, listener, planSegments(text, budget))
        return try {
            rewriteOnce(styleToken, runner, text, listener)
        } catch (overflow: RewriteException.OutputTooLong) {
            val halves = splitAfterOverflow(text, tokens, overflow)
            listener?.onRestart()
            rewriteChunked(styleToken, runner, listener, halves)
        }
    }

    private suspend fun <M, C> rewriteOnce(styleToken: String, runner: Seq2SeqRunner<M, C>, text: String, listener: RewriteListener?): RewriteResult {
        val inputIds = promptIds(styleToken, text)
        val result = generateIds(runner, inputIds, listener)
        if (!result.finished) throw RewriteException.OutputTooLong(base.limits.maxNewTokens)
        val output = decode(result.ids).trim()
        if (output.isEmpty()) throw RewriteException.EmptyResult()
        return RewriteResult(output, inputIds.size, result.ids.size)
    }

    /** Tokens [text] alone would add to the prompt, exactly as the frame would encode it. */
    private fun measureTokens(text: String): Int = if (text.isEmpty()) 0 else prompts.encodeInput(text).size

    /**
     * The most raw-text tokens one call takes: what the frame has room for, and no more than a full-ratio rewrite
     * of it fits in `maxNewTokens`. Frame overhead is measured from the frame itself (an empty prompt build).
     */
    private fun chunkBudget(style: Style, styleToken: String): Int {
        val overhead = promptIds(styleToken, "").size
        val inputLimit = max(1, base.limits.maxInputTokens - overhead)
        val outputBudget = max(1, floor(base.limits.maxNewTokens / style.maxExpansion).toInt())
        return min(inputLimit, outputBudget)
    }

    /**
     * [text] as calls of at most [budget] tokens and the separators between them. Pieces are packed into as few
     * calls as fit, cut near equal size so the last call is not a stray sentence rewritten without its context; a
     * piece too big on its own is cut at the next finer split.
     */
    private fun planSegments(text: String, budget: Int, level: Int = 0): List<Segment> {
        val total = measureTokens(text)
        if (total <= budget) return if (text.isEmpty()) emptyList() else listOf(Segment.Chunk(text))
        val split = SPLITS.getOrNull(level) ?: throw RewriteException.InputTooLong(total, budget)
        val target = ceil(total.toDouble() / ceil(total.toDouble() / budget)).toInt()
        val parts = splitKeepingSeparators(text, split) // pieces at even indexes, the separators between them at odd ones
        val segments = ArrayList<Segment>()
        var current = ""
        var gap = ""
        fun close() {
            if (current.isNotEmpty()) segments.add(Segment.Chunk(current))
            current = ""
        }
        var index = 0
        while (index < parts.size) {
            val piece = parts[index]
            if (piece.isNotEmpty()) {
                if (current.isNotEmpty() && (measureTokens(current) >= target || measureTokens(current + gap + piece) > budget)) close()
                if (current.isNotEmpty()) {
                    current += gap + piece
                } else {
                    if (gap.isNotEmpty()) segments.add(Segment.Join(gap))
                    if (measureTokens(piece) <= budget) current = piece else segments.addAll(planSegments(piece, budget, level + 1))
                }
                gap = ""
            }
            gap += parts.getOrElse(index + 1) { "" }
            index += 2
        }
        close()
        if (gap.isNotEmpty()) segments.add(Segment.Join(gap))
        return segments
    }

    /** A call that ran past maxNewTokens, as calls of half its size; [overflow] is rethrown when it cannot be cut. */
    private fun splitAfterOverflow(text: String, tokens: Int, overflow: RewriteException.OutputTooLong): List<Segment> {
        if (tokens < 2) throw overflow
        val segments = try {
            // Any piece smaller than the whole is allowed; the near-equal cut still puts it close to the middle.
            planSegments(text, tokens - 1)
        } catch (_: RewriteException.InputTooLong) {
            throw overflow
        }
        if (segments.count { it is Segment.Chunk } < 2) throw overflow
        return segments.map { if (it is Segment.Chunk) it.copy(halved = true) else it }
    }

    /**
     * Rewrite each chunk in order, streaming the joined text so far through the listener. A chunk whose rewrite runs
     * past maxNewTokens is redone as two halves (once); the chunks before it are kept.
     */
    private suspend fun <M, C> rewriteChunked(styleToken: String, runner: Seq2SeqRunner<M, C>, listener: RewriteListener?, planned: List<Segment>): RewriteResult {
        val segments = planned.toMutableList()
        // Build every chunk's prompt before the first model call, so text the prompt builder refuses fails at once,
        // as it does for a short text, instead of after the earlier chunks were rewritten.
        segments.filterIsInstance<Segment.Chunk>().forEach { promptIds(styleToken, it.text) }
        var completed = ""
        var inputTokens = 0
        var outputTokens = 0
        var index = 0
        while (index < segments.size) {
            coroutineContext.ensureActive()
            val segment = segments[index]
            if (segment is Segment.Join) {
                completed += segment.text
                index++
                continue
            }
            segment as Segment.Chunk
            val prefix = completed
            val chunkListener = listener?.let { outer ->
                object : RewriteListener {
                    override fun onToken(count: Int, readText: () -> String) = outer.onToken(count) { prefix + readText() }
                }
            }
            val result = try {
                rewriteOnce(styleToken, runner, segment.text, chunkListener)
            } catch (overflow: RewriteException.OutputTooLong) {
                if (segment.halved) throw overflow
                segments.removeAt(index)
                segments.addAll(index, splitAfterOverflow(segment.text, measureTokens(segment.text), overflow))
                continue
            }
            completed = prefix + result.text
            inputTokens += result.inputTokens
            outputTokens += result.outputTokens
            index++
        }
        val output = completed.trim()
        if (output.isEmpty()) throw RewriteException.EmptyResult()
        return RewriteResult(output, inputTokens, outputTokens)
    }

    internal companion object {
        /**
         * How a text too long for one call is cut: paragraphs, then sentences, then words. Each split keeps its
         * separator, so the original line breaks and spacing come back unchanged between the rewritten pieces.
         */
        val SPLITS = listOf(Regex("""\n{2,}"""), Regex("""(?U)(?<=[.!?])\s+"""), Regex("""(?U)\s+"""))

        /** JavaScript's `split` with a capturing pattern: pieces at even indexes, the separators at odd ones. */
        fun splitKeepingSeparators(text: String, separator: Regex): List<String> {
            val parts = ArrayList<String>()
            var last = 0
            for (match in separator.findAll(text)) {
                parts.add(text.substring(last, match.range.first))
                parts.add(match.value)
                last = match.range.last + 1
            }
            parts.add(text.substring(last))
            return parts
        }
    }
}
