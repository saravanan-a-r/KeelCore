package com.echomebetter.engine

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import org.junit.AfterClass
import org.junit.BeforeClass
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * The whole rewrite path (prompt assembly, onnxruntime with a LoRA adapter active, the KV-cached greedy loop,
 * logits processors, decoding) against the reference outputs the extension is held to: a tiny random
 * encoder-decoder in Thelivu's package format, with repetition penalty and no-repeat n-gram switched on. Every case
 * also records the plain base model's output, which differs: an adapter that silently failed to apply shows here.
 */
class ParityTest {
    companion object {
        private lateinit var model: Fixtures.Model
        private lateinit var int8: Fixtures.Model

        @BeforeClass
        @JvmStatic
        fun open() {
            model = Fixtures.Model(Fixtures.dir("tiny-echo"))
            int8 = Fixtures.Model(Fixtures.dir("tiny-echo-int8"))
        }

        @AfterClass
        @JvmStatic
        fun close() {
            model.close()
            int8.close()
        }
    }

    private fun withoutEos(m: Fixtures.Model, ids: List<Int>) = if (ids.lastOrNull() == m.index.base.tokens.eosId) ids.dropLast(1) else ids

    private fun styleOf(case: Fixtures.GenerationCase) = requireNotNull(Style.fromId(case.style)) { "fixture has unknown style ${case.style}" }

    @Test
    fun `prompt, adapted and plain generation match the reference ids`() = runBlocking {
        val cases = Fixtures.parity("tiny-echo")
        check(cases.isNotEmpty())
        for (case in cases) {
            val adapterStyle = model.adapterStyle(styleOf(case))
            val styleToken = model.adapters.getValue(adapterStyle).styleToken
            assertEquals(case.ids, model.engine.promptIds(styleToken, case.input), "prompt of ${case.style}: ${case.input}")
            model.loadAdapter(adapterStyle).use { adapter ->
                val adapted = model.engine.generateIds(model.baseRunner.withAdapter(adapter), case.ids)
                assertEquals(withoutEos(model, case.outputIds), adapted.ids, "adapted output of ${case.style}: ${case.input}")
            }
            val plain = model.engine.generateIds(model.baseRunner, case.ids)
            assertEquals(withoutEos(model, case.baseOutputIds), plain.ids, "base output of ${case.style}: ${case.input}")
        }
    }

    @Test
    fun `int8 base graphs run with adapters, and the adapter changes the output`() = runBlocking {
        val vocab = int8.index.base.architecture.vocabSize
        for (case in Fixtures.parity("tiny-echo-int8")) {
            val adapterStyle = int8.adapterStyle(styleOf(case))
            assertEquals(case.ids, int8.engine.promptIds(int8.adapters.getValue(adapterStyle).styleToken, case.input))
            val adapted = int8.loadAdapter(adapterStyle).use { int8.engine.generateIds(int8.baseRunner.withAdapter(it), case.ids) }
            assertTrue(adapted.ids.size <= case.maxNewTokens)
            assertTrue(adapted.ids.all { it in 0 until vocab })
            assertNotEquals(int8.engine.generateIds(int8.baseRunner, case.ids).ids, adapted.ids)
        }
    }

    @Test
    fun `rewrite returns the decoded text of a finished generation and streams the same text`() = runBlocking {
        val case = Fixtures.parity("tiny-echo").first { it.outputIds.lastOrNull() == model.index.base.tokens.eosId }
        val style = styleOf(case)
        val adapterStyle = model.adapterStyle(style)
        val styleToken = model.adapters.getValue(adapterStyle).styleToken
        model.loadAdapter(adapterStyle).use { adapter ->
            val runner = model.baseRunner.withAdapter(adapter)
            val result = model.engine.rewrite(style, styleToken, runner, case.input)
            assertEquals(case.outputText.trim(), result.text)

            val sofar = ArrayList<String>()
            val streamed = model.engine.rewrite(style, styleToken, runner, case.input, object : RewriteListener {
                override fun onToken(count: Int, readText: () -> String) {
                    sofar.add(readText())
                }
            })
            assertEquals(result.text, streamed.text)
            assertEquals(result.outputTokens, sofar.size)
            assertEquals(result.text, sofar.last())
        }
    }

    @Test
    fun `an adapter released after one rewrite can be loaded again, and the base keeps working`() = runBlocking {
        val case = Fixtures.parity("tiny-echo").first()
        val adapterStyle = model.adapterStyle(styleOf(case))
        val first = model.loadAdapter(adapterStyle).use { model.engine.generateIds(model.baseRunner.withAdapter(it), case.ids).ids }
        val again = model.loadAdapter(adapterStyle).use { model.engine.generateIds(model.baseRunner.withAdapter(it), case.ids).ids }
        assertEquals(first, again)
        assertEquals(withoutEos(model, case.baseOutputIds), model.engine.generateIds(model.baseRunner, case.ids).ids)
    }

    @Test
    fun `generation stops at the next step once the caller is cancelled`() {
        val steps = ArrayList<Int>()
        assertFailsWith<CancellationException> {
            runBlocking {
                model.engine.generateIds(model.baseRunner, Fixtures.parity("tiny-echo")[1].ids, object : RewriteListener {
                    override fun onToken(count: Int, readText: () -> String) {
                        steps.add(count)
                        if (count == 2) coroutineContext[Job]!!.cancel()
                    }
                })
            }
        }
        assertEquals(listOf(1, 2), steps)
    }

    @Test
    fun `a rewrite cut off by maxNewTokens is refused rather than returned truncated`() = runBlocking {
        val case = Fixtures.parity("tiny-echo").first { it.outputIds.lastOrNull() != model.index.base.tokens.eosId }
        val style = styleOf(case)
        val adapterStyle = model.adapterStyle(style)
        model.loadAdapter(adapterStyle).use { adapter ->
            assertFailsWith<RewriteException.OutputTooLong> {
                model.engine.rewrite(style, model.adapters.getValue(adapterStyle).styleToken, model.baseRunner.withAdapter(adapter), case.input)
            }
        }
        Unit
    }
}
