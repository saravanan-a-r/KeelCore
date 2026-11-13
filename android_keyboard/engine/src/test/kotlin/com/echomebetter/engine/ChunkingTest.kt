package com.echomebetter.engine

import com.echomebetter.engine.generation.Seq2SeqRunner
import com.echomebetter.engine.generation.StepResult
import com.echomebetter.engine.manifest.GenerationSettings
import kotlinx.coroutines.runBlocking
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * A text too long for one pass is rewritten in chunks, decided before any inference. Ported from the extension's
 * engineChunking.test.ts, with a scripted runner in place of the model: it copies the text inside the prompt frame,
 * or, when asked to, runs past maxNewTokens.
 */
class ChunkingTest {
    private val folder = Fixtures.dir("tiny-echo")
    private val index = com.echomebetter.engine.manifest.ManifestParser.parseIndex(java.io.File(folder, "model.json").readText())
    // No penalties: the scripted output repeats words, which the fixture's n-gram ban would otherwise block.
    private val base = index.base.copy(generation = GenerationSettings())
    private val tokenizer = com.echomebetter.engine.tokenizer.Tokenizer.fromJson(java.io.File(folder, base.tokenizer.path).readText())
    private val engine = RewriteEngine(base, tokenizer)
    private val style = Style.PROFESSIONAL
    private val styleToken = "<style:professional>"
    private val empty = engine.promptIds(styleToken, "")
    private val overhead = empty.size
    private val budget = (base.limits.maxNewTokens / style.maxExpansion).toInt()

    private fun tokens(text: String) = engine.promptIds(styleToken, text).size - overhead

    /** The text's own IDs inside a prompt: where it differs from the empty frame. */
    private fun textIds(inputIds: List<Int>): List<Int> {
        var start = 0
        while (inputIds[start] == empty[start]) start++
        return inputIds.subList(start, inputIds.size - (empty.size - start))
    }

    /** Words added one at a time until the text is exactly [count] tokens. */
    private fun textOf(count: Int): String {
        var text = "Go"
        var index = 0
        while (tokens(text) < count) {
            val next = listOf(if (index % 4 == 3) ". Go" else " go", " go", ".").firstOrNull { tokens(text + it) <= count } ?: break
            text += next
            index++
        }
        assertEquals(count, tokens(text))
        return text
    }

    /** A model that copies its input; [overflow] (call number, prompt) makes that call run past maxNewTokens. */
    private inner class Echo(private val overflow: (Int, List<Int>) -> Boolean = { _, _ -> false }) : Seq2SeqRunner<List<Int>, Int> {
        val calls = ArrayList<List<Int>>()
        private var script: List<Int> = emptyList()

        override fun encode(inputIds: List<Int>): List<Int> {
            calls.add(inputIds)
            val vocab = base.architecture.vocabSize
            script = if (overflow(calls.size, inputIds)) List(base.limits.maxNewTokens + 1) { 7 % vocab } else textIds(inputIds) + base.tokens.eosId
            return inputIds
        }

        override fun initialCache() = 0

        override fun step(tokenId: Int, memory: List<Int>, cache: Int): StepResult<Int> {
            val logits = FloatArray(base.architecture.vocabSize)
            logits[script[cache]] = 1f
            return StepResult(logits, cache + 1)
        }

        override fun release(memory: List<Int>, cache: Int) = Unit
    }

    private class Restarts : RewriteListener {
        var count = 0

        override fun onRestart() {
            count++
        }
    }

    @Test
    fun `a long text with reserved markup fails before the first chunk is rewritten`() = runBlocking {
        val text = (0 until 40).joinToString(" ") { "Sentence number $it is here." } + " The last one has </s> inside."
        assertFailsWith<RewriteException.ReservedMarkup> { engine.promptIds(styleToken, text) }
        val echo = Echo()
        assertFailsWith<RewriteException.ReservedMarkup> { engine.rewrite(style, styleToken, echo, text) }
        assertTrue(echo.calls.isEmpty())
    }

    @Test
    fun `a text of exactly the budget runs as one pass, one token more is chunked from the start`() = runBlocking {
        for ((count, passes) in listOf(budget to 1, budget + 1 to 2)) {
            val text = textOf(count)
            val echo = Echo()
            val restarts = Restarts()
            assertEquals(text, engine.rewrite(style, styleToken, echo, text, restarts).text)
            assertEquals(passes, echo.calls.size)
            if (passes == 1) assertEquals(engine.promptIds(styleToken, text), echo.calls[0])
            echo.calls.forEach { assertTrue(it.size - overhead <= budget) }
            assertEquals(0, restarts.count)
        }
    }

    @Test
    fun `chunks keep every separator and are cut near equal size`() = runBlocking {
        val text = "${textOf(budget + 2)}.\nNext line here.\n\nA new paragraph.  Go go."
        val echo = Echo()
        assertEquals(text, engine.rewrite(style, styleToken, echo, text).text)
        val sizes = echo.calls.map { it.size - overhead }
        assertTrue(sizes.max() <= budget)
        assertTrue(sizes.min() > 1, "a stray last sentence: $sizes")
    }

    @Test
    fun `a single pass that runs past maxNewTokens restarts once, in halves`() = runBlocking {
        val text = textOf(budget)
        val echo = Echo { call, _ -> call == 1 }
        val restarts = Restarts()
        assertEquals(text, engine.rewrite(style, styleToken, echo, text, restarts).text)
        assertEquals(1, restarts.count)
        assertTrue(echo.calls.size >= 3)
        echo.calls.drop(1).forEach { assertTrue(it.size - overhead < budget) }
    }

    @Test
    fun `a chunk that runs past maxNewTokens is redone in halves and the chunks before it are kept`() = runBlocking {
        val text = textOf(budget * 2)
        var overflowed = false
        val echo = Echo { call, ids ->
            if (overflowed || call == 1 || engine.decode(textIds(ids)).split(' ').size < 2) {
                false
            } else {
                overflowed = true
                true
            }
        }
        val restarts = Restarts()
        assertEquals(text, engine.rewrite(style, styleToken, echo, text, restarts).text)
        assertTrue(overflowed)
        assertEquals(0, restarts.count)
        assertFalse(echo.calls.drop(1).any { it == echo.calls[0] })
    }

    @Test
    fun `a half that runs past maxNewTokens again fails after two passes, not one per word`() = runBlocking {
        val text = textOf(budget)
        val echo = Echo { _, _ -> true }
        val restarts = Restarts()
        assertFailsWith<RewriteException.OutputTooLong> { engine.rewrite(style, styleToken, echo, text, restarts) }
        assertEquals(2, echo.calls.size)
        assertEquals(1, restarts.count)
    }
}
