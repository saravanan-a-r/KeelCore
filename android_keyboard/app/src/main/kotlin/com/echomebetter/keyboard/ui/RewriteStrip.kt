package com.echomebetter.keyboard.ui

import androidx.compose.foundation.basicMarquee
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.echomebetter.engine.Style
import com.echomebetter.keyboard.RewriteAvailability
import dev.patrickgold.florisboard.rewriteSession
import com.echomebetter.rewrite.R
import com.echomebetter.rewrite.editor.RewriteMessage
import com.echomebetter.rewrite.editor.RewriteState
import com.echomebetter.rewrite.label
import dev.patrickgold.florisboard.ime.smartbar.CandidatesRow
import dev.patrickgold.florisboard.ime.theme.FlorisImeUi
import dev.patrickgold.florisboard.nlpManager
import org.florisboard.lib.snygg.SnyggSelector
import org.florisboard.lib.snygg.ui.SnyggIcon
import org.florisboard.lib.snygg.ui.SnyggRow
import org.florisboard.lib.snygg.ui.SnyggSpacer
import org.florisboard.lib.snygg.ui.SnyggText
import org.florisboard.lib.snygg.ui.rememberSnyggThemeQuery
import java.text.NumberFormat

/**
 * The Rewrite strip: shown in the Smartbar where word suggestions go, whenever there are none. It is drawn with the
 * Smartbar's own suggestion elements (`smartbar-candidate-word`, `-text`, `-spacer`), so every FlorisBoard theme,
 * including the user's own, styles it like the rest of the bar.
 */
@Composable
fun RewriteStrip(modifier: Modifier = Modifier) {
    val session by LocalContext.current.rewriteSession()
    val state by session.state.collectAsState()
    val availability by session.availability.collectAsState(RewriteAvailability.CHECKING)
    val style by session.style.collectAsState(Style.GRAMMAR)

    SnyggRow(
        elementName = FlorisImeUi.SmartbarCandidatesRow.elementName,
        modifier = modifier.fillMaxSize(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Start,
    ) {
        when {
            availability == RewriteAvailability.PASSWORD_FIELD -> Unit
            availability == RewriteAvailability.NEEDS_SETUP -> {
                StripItem(Icons.Default.AutoAwesome, stringResource(R.string.strip_set_up), onClick = session::openSettings)
            }
            else -> when (val current = state) {
                RewriteState.Idle -> {
                    StripItem(
                        Icons.Default.AutoAwesome,
                        stringResource(R.string.strip_rewrite),
                        enabled = availability == RewriteAvailability.READY,
                        onClick = session::rewrite,
                    )
                    Divider()
                    StyleChip(style, onClick = session::openStylePanel)
                }
                is RewriteState.Working -> Working(current, onCancel = session.controller::cancel)
                is RewriteState.Done -> {
                    StripItem(
                        if (current.canUndo) Icons.AutoMirrored.Filled.Undo else Icons.Default.Check,
                        stringResource(
                            when {
                                current.canUndo -> R.string.strip_undo
                                current.unchanged -> R.string.strip_unchanged
                                else -> R.string.strip_done
                            },
                        ),
                        enabled = current.canUndo,
                        onClick = session.controller::undo,
                    )
                    Divider()
                    // The chip names the style, and another style rewrites the same original text.
                    StyleChip(style, onClick = session::openStylePanel)
                }
                is RewriteState.Failed -> {
                    StripText(messageText(current.message), Modifier.weight(1f))
                    StripIcon(Icons.Default.Close, stringResource(R.string.strip_dismiss), onClick = session.controller::onTyping)
                }
            }
        }
    }
}

@Composable
private fun RowScope.Working(state: RewriteState.Working, onCancel: () -> Unit) {
    val color = rememberSnyggThemeQuery(FlorisImeUi.SmartbarCandidateWordText.elementName).foreground()
    CircularProgressIndicator(
        modifier = Modifier
            .padding(start = 12.dp, end = 4.dp)
            .size(18.dp),
        color = color,
        strokeWidth = 2.dp,
    )
    val text = when {
        state.loading -> stringResource(R.string.strip_loading)
        // The newest words matter while it writes: a long preview shows its end.
        state.preview.isNotEmpty() -> if (state.preview.length > PREVIEW_CHARS) "…" + state.preview.takeLast(PREVIEW_CHARS) else state.preview
        else -> stringResource(R.string.strip_writing)
    }
    StripText(text, Modifier.weight(1f))
    StripIcon(Icons.Default.Close, stringResource(R.string.strip_cancel), onClick = onCancel)
}

/** One tappable item, styled as a suggestion: an icon and a label. */
@Composable
private fun StripItem(icon: ImageVector, label: String, enabled: Boolean = true, onClick: () -> Unit) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    SnyggRow(
        elementName = FlorisImeUi.SmartbarCandidateWord.elementName,
        selector = if (pressed) SnyggSelector.PRESSED else SnyggSelector.NONE,
        modifier = Modifier
            .fillMaxHeight()
            .alpha(if (enabled) 1f else DISABLED_ALPHA),
        clickAndSemanticsModifier = Modifier.clickable(
            interactionSource = interaction,
            indication = null,
            enabled = enabled,
            role = Role.Button,
            onClick = onClick,
        ),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        SnyggIcon(modifier = Modifier.size(18.dp), imageVector = icon)
        SnyggText(
            elementName = FlorisImeUi.SmartbarCandidateWordText.elementName,
            modifier = Modifier.padding(start = 6.dp),
            text = label,
        )
    }
}

/** The style chip: the chosen style's name and a chevron that opens the style panel. */
@Composable
private fun StyleChip(style: Style, onClick: () -> Unit) {
    val context = LocalContext.current
    val name = context.getString(style.label)
    val description = stringResource(R.string.strip_style_description, name)
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    SnyggRow(
        elementName = FlorisImeUi.SmartbarCandidateWord.elementName,
        selector = if (pressed) SnyggSelector.PRESSED else SnyggSelector.NONE,
        modifier = Modifier
            .fillMaxHeight()
            .semantics { contentDescription = description },
        clickAndSemanticsModifier = Modifier.clickable(interactionSource = interaction, indication = null, role = Role.Button, onClick = onClick),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        SnyggText(elementName = FlorisImeUi.SmartbarCandidateWordSecondaryText.elementName, text = name)
        SnyggIcon(modifier = Modifier.size(18.dp), imageVector = Icons.Default.ExpandMore)
    }
}

@Composable
private fun StripText(text: String, modifier: Modifier) {
    Row(
        modifier = modifier
            .padding(horizontal = 8.dp)
            .semantics { liveRegion = LiveRegionMode.Polite },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        SnyggText(
            elementName = FlorisImeUi.SmartbarCandidateWordText.elementName,
            modifier = Modifier
                .fillMaxWidth()
                .basicMarquee(),
            text = text,
        )
    }
}

@Composable
private fun StripIcon(icon: ImageVector, description: String, onClick: () -> Unit) {
    SnyggRow(
        elementName = FlorisImeUi.SmartbarCandidateWord.elementName,
        modifier = Modifier
            .fillMaxHeight()
            .width(48.dp),
        clickAndSemanticsModifier = Modifier.clickable(role = Role.Button, onClickLabel = description, onClick = onClick),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
    ) {
        SnyggIcon(modifier = Modifier.size(20.dp), imageVector = icon, contentDescription = description)
    }
}

@Composable
private fun Divider() {
    SnyggSpacer(
        elementName = FlorisImeUi.SmartbarCandidateSpacer.elementName,
        modifier = Modifier
            .width(1.dp)
            .fillMaxHeight(0.6f),
    )
}

@Composable
private fun messageText(message: RewriteMessage): String = when (message) {
    RewriteMessage.NothingToRewrite -> stringResource(R.string.message_nothing_to_rewrite)
    is RewriteMessage.TooLong -> stringResource(R.string.message_too_long, NumberFormat.getIntegerInstance().format(message.limit))
    RewriteMessage.InputTooLong -> stringResource(R.string.message_input_too_long)
    RewriteMessage.OutputTooLong -> stringResource(R.string.message_output_too_long)
    RewriteMessage.EmptyResult -> stringResource(R.string.message_empty_result)
    RewriteMessage.ReservedMarkup -> stringResource(R.string.message_reserved_markup)
    RewriteMessage.TextChanged -> stringResource(R.string.message_text_changed)
    RewriteMessage.UnsupportedField -> stringResource(R.string.message_unsupported_field)
    RewriteMessage.LoadFailed -> stringResource(R.string.message_load_failed)
    is RewriteMessage.NotInstalled -> stringResource(R.string.message_style_not_installed, LocalContext.current.getString(message.style.label))
}

private const val PREVIEW_CHARS = 48
private const val DISABLED_ALPHA = 0.5f

/** The Smartbar's suggestion area: FlorisBoard's suggestions when it has any, else the Rewrite strip. */
@Composable
fun CandidatesOrRewriteStrip(modifier: Modifier = Modifier) {
    val nlpManager by LocalContext.current.nlpManager()
    val candidates by nlpManager.activeCandidatesFlow.collectAsState()
    if (candidates.isEmpty()) RewriteStrip(modifier) else CandidatesRow(modifier)
}
