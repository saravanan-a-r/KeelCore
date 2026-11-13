package com.echomebetter.rewrite.editor

import android.view.inputmethod.InputConnection

/** The editor's selection; [start] == [end] is a plain cursor. */
data class Selection(val start: Int, val end: Int)

/**
 * A run of the field's text, by absolute position. [wholeField]: the run was the entire field, so it is still the
 * target only while the field holds nothing else (text typed after it makes it a different target).
 */
data class TextRange(val start: Int, val end: Int, val text: String, val wholeField: Boolean = false)

/** What the Rewrite button would work on. */
sealed interface Capture {
    data class Target(val range: TextRange) : Capture

    /** The field (or the selection) is empty or only spaces. */
    data object Empty : Capture

    /** More than [limit] characters. */
    data class TooLong(val limit: Int) : Capture

    /** The app does not expose its text to the keyboard. */
    data object Unsupported : Capture
}

/** The field being edited, as the rewrite needs it. */
interface EditableField {
    /** What the Rewrite button would work on: the selection, or else the whole field. */
    fun capture(): Capture

    /** The field's text at [start, end), or null when it cannot be read. */
    fun read(start: Int, end: Int): String?

    /** Replaces [range] with [text] if the field still holds [range]'s text there; false (and nothing done) if not. */
    fun replace(range: TextRange, text: String): Boolean
}

/**
 * Reads and replaces the text of the field being edited, through its [InputConnection].
 *
 * Positions come from the selection the editor reports to the keyboard ([selection]), and the text around it from
 * `getTextBeforeCursor` / `getSelectedText` / `getTextAfterCursor`, which every editor implements. With a selection the
 * selection is the target; without one, the whole field. Replacing first checks that the text is still what was
 * read, so a rewrite never overwrites something the user changed meanwhile.
 */
class FieldEditor(
    private val connection: () -> InputConnection?,
    private val selection: () -> Selection?,
    private val maxChars: Int = MAX_CHARS,
) : EditableField {
    override fun capture(): Capture {
        val ic = connection() ?: return Capture.Unsupported
        val (start, end) = selection() ?: return Capture.Unsupported
        if (start != end) {
            if (end - start > maxChars) return Capture.TooLong(maxChars)
            val text = ic.getSelectedText(0)?.toString() ?: return Capture.Unsupported
            if (text.length != end - start) return Capture.Unsupported
            return if (text.isBlank()) Capture.Empty else Capture.Target(TextRange(start, end, text))
        }
        val field = readField(ic) ?: return Capture.Unsupported
        if (field.text.length > maxChars) return Capture.TooLong(maxChars)
        return if (field.text.isBlank()) Capture.Empty else Capture.Target(field)
    }

    override fun read(start: Int, end: Int): String? {
        val ic = connection() ?: return null
        val selected = selection() ?: return null
        val before = ic.getTextBeforeCursor(maxOf(selected.start - start, 0), 0)?.toString() ?: return null
        val middle = if (selected.start != selected.end) ic.getSelectedText(0)?.toString() ?: return null else ""
        val after = ic.getTextAfterCursor(maxOf(end - selected.end, 0), 0)?.toString() ?: return null
        val known = TextRange(selected.start - before.length, selected.end + after.length, before + middle + after)
        if (start < known.start || end > known.end) return null
        return known.text.substring(start - known.start, end - known.start)
    }

    override fun replace(range: TextRange, text: String): Boolean {
        val ic = connection() ?: return false
        val unchanged = if (range.wholeField) readField(ic) == range else read(range.start, range.end) == range.text
        if (!unchanged) return false
        ic.beginBatchEdit()
        try {
            ic.finishComposingText()
            ic.setSelection(range.start, range.end)
            ic.commitText(text, 1)
        } finally {
            ic.endBatchEdit()
        }
        return true
    }

    /** The whole field (up to one character past the limit, so an over-long field shows as such). */
    private fun readField(ic: InputConnection): TextRange? {
        val (start, end) = selection() ?: return null
        val limit = maxChars + 1
        val before = ic.getTextBeforeCursor(limit, 0)?.toString() ?: return null
        val selected = if (start != end) ic.getSelectedText(0)?.toString() ?: return null else ""
        val after = ic.getTextAfterCursor(limit, 0)?.toString() ?: return null
        return TextRange(start - before.length, end + after.length, before + selected + after, wholeField = true)
    }

    companion object {
        /** About 1,000 words: what one rewrite should take on a phone. */
        const val MAX_CHARS = 6_000
    }
}
