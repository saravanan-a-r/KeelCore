package com.echomebetter.rewrite.editor

import com.echomebetter.engine.RewriteException
import com.echomebetter.engine.RewriteResult
import com.echomebetter.engine.Style
import com.echomebetter.rewrite.engine.ModelUnavailableException
import com.echomebetter.rewrite.engine.RewritePhase
import com.echomebetter.rewrite.engine.Rewriter
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class RewriteControllerTest {
    /** A field in memory, with the cursor at the end unless a selection is set. */
    private class FakeField(var text: String, var selection: IntRange? = null) : EditableField {
        override fun capture(): Capture {
            val range = selection?.let { TextRange(it.first, it.last, text.substring(it.first, it.last)) } ?: TextRange(0, text.length, text)
            return if (range.text.isBlank()) Capture.Empty else Capture.Target(range)
        }

        override fun read(start: Int, end: Int): String? = if (start in 0..end && end <= text.length) text.substring(start, end) else null

        override fun replace(range: TextRange, text: String): Boolean {
            if (read(range.start, range.end) != range.text) return false
            this.text = this.text.replaceRange(range.start, range.end, text)
            return true
        }
    }

    /** Answers each style with a fixed rewrite of whatever it is given, and records the inputs. */
    private class FakeRewriter(private val answer: (Style, String) -> String) : Rewriter {
        val inputs = ArrayList<Pair<Style, String>>()
        var failure: Exception? = null
        var gate: CompletableDeferred<Unit>? = null

        override suspend fun rewrite(style: Style, text: String, onPhase: (RewritePhase) -> Unit): RewriteResult {
            inputs.add(style to text)
            onPhase(RewritePhase.Loading)
            gate?.await()
            failure?.let { throw it }
            val output = answer(style, text)
            onPhase(RewritePhase.Writing { output })
            return RewriteResult(output, 1, 1)
        }
    }

    private val rewriter = FakeRewriter { style, text -> "[${style.id}] $text" }

    private fun TestScope.controller(field: FakeField) = RewriteController(field, rewriter, this, messageMs = 1_000)

    @Test
    fun `a rewrite replaces the field and Undo puts the original back`() = runTest {
        val field = FakeField("hello there")
        val controller = controller(field)
        controller.rewrite(Style.GRAMMAR)
        advanceUntilIdle()
        assertEquals("[grammar] hello there", field.text)
        assertEquals(RewriteState.Done(Style.GRAMMAR, unchanged = false, canUndo = true), controller.state.value)
        controller.undo()
        assertEquals("hello there", field.text)
        assertEquals(RewriteState.Idle, controller.state.value)
    }

    @Test
    fun `another style rewrites the original text, not the last rewrite`() = runTest {
        val field = FakeField("hello there")
        val controller = controller(field)
        controller.rewrite(Style.GRAMMAR)
        advanceUntilIdle()
        controller.rewrite(Style.FRIENDLY)
        advanceUntilIdle()
        assertEquals(listOf(Style.GRAMMAR to "hello there", Style.FRIENDLY to "hello there"), rewriter.inputs)
        assertEquals("[friendly] hello there", field.text)
        controller.undo()
        assertEquals("hello there", field.text)
    }

    @Test
    fun `typing makes the rewrite the user's text`() = runTest {
        val field = FakeField("hello there")
        val controller = controller(field)
        controller.rewrite(Style.GRAMMAR)
        advanceUntilIdle()
        field.text += "!"
        controller.onTyping()
        controller.rewrite(Style.CONCISE)
        advanceUntilIdle()
        assertEquals(Style.CONCISE to "[grammar] hello there!", rewriter.inputs.last())
    }

    @Test
    fun `spaces around the text stay where they were, and the model sees the text without them`() = runTest {
        val field = FakeField("  hello there \n")
        controller(field).rewrite(Style.GRAMMAR)
        advanceUntilIdle()
        assertEquals("hello there", rewriter.inputs.single().second)
        assertEquals("  [grammar] hello there \n", field.text)
    }

    @Test
    fun `a selection is rewritten in place`() = runTest {
        val field = FakeField("Keep. fix me. Keep.", selection = 6..13)
        controller(field).rewrite(Style.GRAMMAR)
        advanceUntilIdle()
        assertEquals("Keep. [grammar] fix me. Keep.", field.text)
    }

    @Test
    fun `text changed during the rewrite is left alone`() = runTest {
        val field = FakeField("original")
        rewriter.gate = CompletableDeferred()
        val controller = controller(field)
        controller.rewrite(Style.GRAMMAR)
        runCurrent()
        assertTrue(controller.state.value is RewriteState.Working)
        field.text = "changed meanwhile"
        rewriter.gate!!.complete(Unit)
        runCurrent()
        assertEquals("changed meanwhile", field.text)
        assertEquals(RewriteState.Failed(RewriteMessage.TextChanged), controller.state.value)
    }

    @Test
    fun `cancel stops the rewrite and changes nothing`() = runTest {
        val field = FakeField("original")
        rewriter.gate = CompletableDeferred()
        val controller = controller(field)
        controller.rewrite(Style.GRAMMAR)
        runCurrent()
        assertEquals(RewriteState.Working(Style.GRAMMAR, loading = true, preview = ""), controller.state.value)
        controller.cancel()
        advanceUntilIdle()
        assertEquals("original", field.text)
        assertEquals(RewriteState.Idle, controller.state.value)
    }

    @Test
    fun `an unchanged rewrite says so and offers nothing to undo`() = runTest {
        val field = FakeField("Already fine.")
        val controller = RewriteController(field, FakeRewriter { _, text -> text }, this)
        controller.rewrite(Style.GRAMMAR)
        advanceUntilIdle()
        assertEquals(RewriteState.Done(Style.GRAMMAR, unchanged = true, canUndo = false), controller.state.value)
    }

    @Test
    fun `failures become messages that clear themselves`() = runTest {
        val cases = listOf(
            RewriteException.InputTooLong(10, 5) to RewriteMessage.InputTooLong,
            RewriteException.OutputTooLong(384) to RewriteMessage.OutputTooLong,
            RewriteException.EmptyResult() to RewriteMessage.EmptyResult,
            RewriteException.ReservedMarkup() to RewriteMessage.ReservedMarkup,
            ModelUnavailableException("no memory") to RewriteMessage.LoadFailed,
        )
        for ((error, message) in cases) {
            val field = FakeField("some text")
            rewriter.failure = error
            val controller = controller(field)
            controller.rewrite(Style.GRAMMAR)
            runCurrent()
            assertEquals(RewriteState.Failed(message), controller.state.value)
            assertEquals("some text", field.text)
            advanceUntilIdle()
            assertEquals(RewriteState.Idle, controller.state.value)
        }
    }

    @Test
    fun `an empty field asks for text and calls nothing`() = runTest {
        val controller = controller(FakeField("   "))
        controller.rewrite(Style.GRAMMAR)
        assertEquals(RewriteState.Failed(RewriteMessage.NothingToRewrite), controller.state.value)
        assertTrue(rewriter.inputs.isEmpty())
    }
}
