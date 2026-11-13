package com.echomebetter.rewrite.editor

import com.echomebetter.engine.RewriteException
import com.echomebetter.engine.Style
import com.echomebetter.rewrite.engine.ModelUnavailableException
import com.echomebetter.rewrite.engine.RewritePhase
import com.echomebetter.rewrite.engine.Rewriter
import com.echomebetter.rewrite.engine.StyleNotInstalledException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** Why the last rewrite did not happen, as the bar tells the user. */
sealed interface RewriteMessage {
    data object NothingToRewrite : RewriteMessage
    data class TooLong(val limit: Int) : RewriteMessage
    data object InputTooLong : RewriteMessage
    data object OutputTooLong : RewriteMessage
    data object EmptyResult : RewriteMessage
    data object ReservedMarkup : RewriteMessage
    data object TextChanged : RewriteMessage
    data object UnsupportedField : RewriteMessage
    data object LoadFailed : RewriteMessage
    data class NotInstalled(val style: Style) : RewriteMessage
}

sealed interface RewriteState {
    data object Idle : RewriteState

    /** [preview] is the rewrite so far (empty until the first words). */
    data class Working(val style: Style, val loading: Boolean, val preview: String) : RewriteState

    /** [unchanged]: the model kept the text as it was. [canUndo]: the rewrite is in the field and can be taken back. */
    data class Done(val style: Style, val unchanged: Boolean, val canUndo: Boolean) : RewriteState

    data class Failed(val message: RewriteMessage) : RewriteState
}

/**
 * One rewrite at a time, from the field's text to the rewritten text in the field, with Undo.
 *
 * Rewriting again while the last rewrite is still in the field rewrites the user's original text, not the rewrite:
 * trying one style after another always starts from what the user wrote. Text around the target's ends (spaces, a
 * trailing newline) is kept as it was; the model only sees and replaces what is between.
 */
class RewriteController(
    private val field: EditableField,
    private val rewriter: Rewriter,
    private val scope: CoroutineScope,
    private val messageMs: Long = 4_000,
) {
    private class Applied(val range: TextRange, val original: String)

    private val _state = MutableStateFlow<RewriteState>(RewriteState.Idle)
    val state: StateFlow<RewriteState> = _state.asStateFlow()

    private var job: Job? = null
    private var dismissJob: Job? = null
    private var applied: Applied? = null

    val isWorking: Boolean get() = job?.isActive == true

    fun rewrite(style: Style) {
        if (isWorking) return
        dismissJob?.cancel()
        val previous = applied?.takeIf(::stillInField)
        val range: TextRange
        val source: String
        if (previous != null) {
            range = previous.range
            source = previous.original
        } else {
            applied = null
            when (val capture = field.capture()) {
                is Capture.Target -> {
                    range = capture.range
                    source = capture.range.text
                }
                Capture.Empty -> return fail(RewriteMessage.NothingToRewrite)
                is Capture.TooLong -> return fail(RewriteMessage.TooLong(capture.limit))
                Capture.Unsupported -> return fail(RewriteMessage.UnsupportedField)
            }
        }
        val lead = source.takeWhile(Char::isWhitespace)
        val trail = source.drop(lead.length).takeLastWhile(Char::isWhitespace)
        val core = source.substring(lead.length, source.length - trail.length)
        _state.value = RewriteState.Working(style, loading = false, preview = "")
        job = scope.launch {
            try {
                val result = rewriter.rewrite(style, core) { phase ->
                    _state.value = when (phase) {
                        RewritePhase.Loading -> RewriteState.Working(style, loading = true, preview = "")
                        is RewritePhase.Writing -> RewriteState.Working(style, loading = false, preview = phase.readText())
                    }
                }
                val text = lead + result.text + trail
                when {
                    text == range.text -> _state.value = RewriteState.Done(style, unchanged = text == source, canUndo = applied != null)
                    field.replace(range, text) -> {
                        applied = Applied(TextRange(range.start, range.start + text.length, text, range.wholeField), source)
                        _state.value = RewriteState.Done(style, unchanged = false, canUndo = true)
                    }
                    else -> fail(RewriteMessage.TextChanged)
                }
            } catch (cancelled: CancellationException) {
                _state.value = RewriteState.Idle
                throw cancelled
            } catch (error: RewriteException.InputTooLong) {
                fail(RewriteMessage.InputTooLong)
            } catch (error: RewriteException.OutputTooLong) {
                fail(RewriteMessage.OutputTooLong)
            } catch (error: RewriteException.EmptyResult) {
                fail(RewriteMessage.EmptyResult)
            } catch (error: RewriteException.ReservedMarkup) {
                fail(RewriteMessage.ReservedMarkup)
            } catch (error: StyleNotInstalledException) {
                fail(RewriteMessage.NotInstalled(error.style))
            } catch (error: ModelUnavailableException) {
                fail(RewriteMessage.LoadFailed)
            }
        }
    }

    /** Whether [rewrite] has something to work on: text in the field, or the last rewrite to redo in another style. */
    fun hasTarget(): Boolean = applied?.let(::stillInField) == true || field.capture() is Capture.Target

    private fun stillInField(last: Applied): Boolean = field.read(last.range.start, last.range.end) == last.range.text

    fun cancel() {
        job?.cancel()
    }

    /** Puts the user's original text back. */
    fun undo() {
        val last = applied ?: return
        if (isWorking) return
        if (field.replace(last.range, last.original)) {
            applied = null
            _state.value = RewriteState.Idle
        } else {
            fail(RewriteMessage.TextChanged)
        }
    }

    /** The user typed: the rewrite is now part of their text, and messages are out of date. */
    fun onTyping() {
        if (isWorking) return
        applied = null
        dismissJob?.cancel()
        _state.value = RewriteState.Idle
    }

    /** Another field, or the keyboard closed: nothing carries over. */
    fun reset() {
        job?.cancel()
        dismissJob?.cancel()
        applied = null
        _state.value = RewriteState.Idle
    }

    private fun fail(message: RewriteMessage) {
        _state.value = RewriteState.Failed(message)
        dismissJob?.cancel()
        dismissJob = scope.launch {
            delay(messageMs)
            if (_state.value == RewriteState.Failed(message)) _state.value = RewriteState.Idle
        }
    }
}
