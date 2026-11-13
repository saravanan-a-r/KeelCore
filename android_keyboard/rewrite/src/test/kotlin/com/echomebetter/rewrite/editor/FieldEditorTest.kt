package com.echomebetter.rewrite.editor

import android.view.inputmethod.EditorInfo
import android.widget.EditText
import androidx.test.core.app.ApplicationProvider
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** The field editor against a real EditText and its own InputConnection. */
@RunWith(RobolectricTestRunner::class)
class FieldEditorTest {
    private val field = EditText(ApplicationProvider.getApplicationContext())
    private val connection = field.onCreateInputConnection(EditorInfo())
    private val editor = FieldEditor({ connection }, { Selection(field.selectionStart, field.selectionEnd) }, maxChars = 100)

    private fun type(text: String, selectionStart: Int = text.length, selectionEnd: Int = selectionStart) {
        field.setText(text)
        field.setSelection(selectionStart, selectionEnd)
    }

    @Test
    fun `without a selection the whole field is the target, wherever the cursor is`() {
        type("hey can u send it", selectionStart = 4)
        assertEquals(Capture.Target(TextRange(0, 17, "hey can u send it", wholeField = true)), editor.capture())
    }

    @Test
    fun `a selection is the target`() {
        type("Keep this. fix thiss one. Keep that.", selectionStart = 11, selectionEnd = 25)
        assertEquals(Capture.Target(TextRange(11, 25, "fix thiss one.")), editor.capture())
    }

    @Test
    fun `empty, blank and over-long fields are reported, not captured`() {
        type("")
        assertEquals(Capture.Empty, editor.capture())
        type("   \n ")
        assertEquals(Capture.Empty, editor.capture())
        type("x".repeat(101))
        assertEquals(Capture.TooLong(100), editor.capture())
        type("x".repeat(101), selectionStart = 0, selectionEnd = 101)
        assertEquals(Capture.TooLong(100), editor.capture())
    }

    @Test
    fun `replace puts the text in place and leaves the cursor after it`() {
        type("say hi to bob tmrw", selectionStart = 3)
        val target = (editor.capture() as Capture.Target).range
        assertTrue(editor.replace(target, "Say hi to Bob tomorrow."))
        assertEquals("Say hi to Bob tomorrow.", field.text.toString())
        assertEquals(23, field.selectionStart)
    }

    @Test
    fun `replace refuses when the field changed since it was read, also by text added after it`() {
        type("first version")
        val target = (editor.capture() as Capture.Target).range
        type("first version, edited")
        assertFalse(editor.replace(target, "Rewritten."))
        assertEquals("first version, edited", field.text.toString())
    }

    @Test
    fun `replace of a selection keeps the text around it`() {
        type("A. b is wrong. C.", selectionStart = 3, selectionEnd = 14)
        val target = (editor.capture() as Capture.Target).range
        assertTrue(editor.replace(target, "B is right."))
        assertEquals("A. B is right. C.", field.text.toString())
    }

    @Test
    fun `read returns any range around the cursor`() {
        type("0123456789", selectionStart = 5)
        assertEquals("234", editor.read(2, 5))
        assertEquals("3456", editor.read(3, 7))
        assertEquals("789", editor.read(7, 10))
    }
}
