package com.echomebetter.keyboard

import android.content.Context
import android.content.Intent
import android.net.Uri
import com.echomebetter.engine.Style
import com.echomebetter.rewrite.editor.FieldEditor
import com.echomebetter.rewrite.editor.RewriteController
import com.echomebetter.rewrite.editor.RewriteState
import com.echomebetter.rewrite.editor.Selection
import com.echomebetter.rewrite.model.InstalledState
import com.echomebetter.rewrite.rewrite
import dev.patrickgold.florisboard.FlorisImeService
import dev.patrickgold.florisboard.app.FlorisAppActivity
import dev.patrickgold.florisboard.editorInstance
import dev.patrickgold.florisboard.ime.ImeUiMode
import dev.patrickgold.florisboard.ime.editor.FlorisEditorInfo
import dev.patrickgold.florisboard.ime.editor.InputAttributes
import dev.patrickgold.florisboard.ime.text.key.KeyCode
import dev.patrickgold.florisboard.keyboardManager
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine

/** Whether the Rewrite button can work in the current field. */
enum class RewriteAvailability {
    READY,

    /** The model is not downloaded: the button opens the Rewrite settings. */
    NEEDS_SETUP,

    /** Never in password fields. */
    PASSWORD_FIELD,

    /** Still reading what is installed (a moment after the keyboard starts). */
    CHECKING,
}

/** What the style panel knows about one style. */
data class StyleOption(val style: Style, val selected: Boolean, val unlocked: Boolean, val installed: Boolean, val free: Boolean)

/**
 * The rewrite feature inside the keyboard: one [RewriteController] working on the field FlorisBoard is editing, the
 * style choice, and the few places the keyboard reports to it (a new field, a key typed). Built once by the
 * application, like FlorisBoard's own managers.
 */
class ImeRewriteSession(context: Context) {
    private val appContext = context.applicationContext
    private val feature = appContext.rewrite
    private val editorInstance by appContext.editorInstance()
    private val keyboardManager by appContext.keyboardManager()

    private val field = FieldEditor(
        connection = { FlorisImeService.currentInputConnection() },
        selection = {
            editorInstance.activeContent.selection.takeIf { it.isValid }?.let { Selection(it.start, it.end) }
        },
    )

    val controller = RewriteController(field, feature.engine, MainScope())

    val state: StateFlow<RewriteState> get() = controller.state

    val availability: Flow<RewriteAvailability> =
        combine(feature.models.installed, editorInstance.activeInfoFlow) { installed, info ->
            when {
                isPassword(info) -> RewriteAvailability.PASSWORD_FIELD
                installed is InstalledState.Checking -> RewriteAvailability.CHECKING
                installed is InstalledState.None -> RewriteAvailability.NEEDS_SETUP
                else -> RewriteAvailability.READY
            }
        }

    /** The chosen style, or the first usable one when the chosen style was removed or is locked. */
    val style: Flow<Style> = combine(feature.settings.style, feature.models.installed) { _, _ -> effectiveStyle() }

    fun rewrite() {
        controller.rewrite(effectiveStyle())
    }

    /** From the style panel: a usable style is chosen and, when there is text, rewritten at once. */
    fun pick(option: StyleOption) {
        if (!option.unlocked || !option.installed) {
            openSettings()
            return
        }
        feature.settings.selectStyle(option.style)
        closeStylePanel()
        if (controller.hasTarget()) controller.rewrite(option.style)
    }

    fun styleOptions(): List<StyleOption> {
        val installed = installedStyles()
        val current = effectiveStyle()
        return Style.entries.map { style ->
            StyleOption(style, style == current, feature.access.isUnlocked(style), style in installed, feature.access.isFree(style))
        }
    }

    fun openStylePanel() {
        keyboardManager.activeState.imeUiMode = ImeUiMode.REWRITE
    }

    fun closeStylePanel() {
        keyboardManager.activeState.imeUiMode = ImeUiMode.TEXT
    }

    /** Hides the keyboard and opens the Rewrite settings screen. */
    fun openSettings() {
        FlorisImeService.hideUi()
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(SETTINGS_DEEPLINK), appContext, FlorisAppActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        appContext.startActivity(intent)
    }

    /** A key went up. Keys that change the text make the last rewrite the user's own text (no more Undo). */
    fun onKeyInput(code: Int) {
        if (code > 0 || code in TEXT_EDITING_KEYS) controller.onTyping()
    }

    /** A swiped word was committed. */
    fun onGestureCommitted() = controller.onTyping()

    /** Another field, or the keyboard was hidden: nothing carries over. */
    fun reset() = controller.reset()

    private fun installedStyles(): Set<Style> =
        (feature.models.installed.value as? InstalledState.Ready)?.model?.manifest?.adapters?.keys.orEmpty()

    private fun effectiveStyle(): Style {
        val chosen = feature.settings.style.value
        val usable = installedStyles().filter(feature.access::isUnlocked)
        return if (chosen in usable || usable.isEmpty()) chosen else usable.first()
    }

    private fun isPassword(info: FlorisEditorInfo): Boolean = info.inputAttributes.variation in PASSWORD_VARIATIONS

    companion object {
        /** The Rewrite settings screen (see Routes.Settings.Rewrite). */
        const val SETTINGS_DEEPLINK = "ui://florisboard/settings/rewrite"

        private val TEXT_EDITING_KEYS = setOf(
            KeyCode.DELETE, KeyCode.DELETE_WORD, KeyCode.FORWARD_DELETE, KeyCode.FORWARD_DELETE_WORD, KeyCode.CLIPBOARD_PASTE, KeyCode.CLIPBOARD_CUT,
        )

        private val PASSWORD_VARIATIONS = setOf(
            InputAttributes.Variation.PASSWORD,
            InputAttributes.Variation.VISIBLE_PASSWORD,
            InputAttributes.Variation.WEB_PASSWORD,
        )
    }
}
