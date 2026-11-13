package com.echomebetter.keyboard.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.sizeIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.echomebetter.keyboard.StyleOption
import dev.patrickgold.florisboard.rewriteSession
import com.echomebetter.rewrite.R
import com.echomebetter.rewrite.hint
import com.echomebetter.rewrite.label
import com.echomebetter.rewrite.rewrite
import dev.patrickgold.florisboard.ime.keyboard.FlorisImeSizing
import dev.patrickgold.florisboard.ime.theme.FlorisImeUi
import org.florisboard.lib.snygg.ui.SnyggBox
import org.florisboard.lib.snygg.ui.SnyggColumn
import org.florisboard.lib.snygg.ui.SnyggIcon
import org.florisboard.lib.snygg.ui.SnyggIconButton
import org.florisboard.lib.snygg.ui.SnyggRow
import org.florisboard.lib.snygg.ui.SnyggText

/**
 * The style list, in the keyboard's place and size (as the clipboard history is). It uses the clipboard panel's
 * theme elements: the same header, content area and item cards, so it looks native in every theme.
 */
@Composable
fun RewriteStylePanel(modifier: Modifier = Modifier) {
    val session by LocalContext.current.rewriteSession()
    // Re-read when the panel opens and whenever the installed styles or the choice change.
    val installed by LocalContext.current.rewrite.models.installed.collectAsState()
    val chosen by session.style.collectAsState(null)
    val options = remember(installed, chosen) { session.styleOptions() }

    SnyggColumn(
        modifier = modifier
            .fillMaxWidth()
            .height(FlorisImeSizing.imeUiHeight()),
    ) {
        SnyggRow(
            elementName = FlorisImeUi.ClipboardHeader.elementName,
            modifier = Modifier
                .fillMaxWidth()
                .height(FlorisImeSizing.smartbarHeight),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            val buttonSize = Modifier.sizeIn(maxHeight = FlorisImeSizing.smartbarHeight).aspectRatio(1f)
            SnyggIconButton(
                elementName = FlorisImeUi.ClipboardHeaderButton.elementName,
                onClick = session::closeStylePanel,
                modifier = buttonSize,
            ) {
                SnyggIcon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.panel_back))
            }
            SnyggText(
                elementName = FlorisImeUi.ClipboardHeaderText.elementName,
                modifier = Modifier
                    .weight(1f)
                    .semantics { heading() },
                text = stringResource(R.string.panel_title),
            )
            SnyggIconButton(
                elementName = FlorisImeUi.ClipboardHeaderButton.elementName,
                onClick = session::openSettings,
                modifier = buttonSize,
            ) {
                SnyggIcon(imageVector = Icons.Outlined.Settings, contentDescription = stringResource(R.string.settings_title))
            }
        }
        SnyggColumn(
            elementName = FlorisImeUi.ClipboardContent.elementName,
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            for (option in options) StyleItem(option, onClick = { session.pick(option) })
        }
    }
}

@Composable
private fun StyleItem(option: StyleOption, onClick: () -> Unit) {
    val context = LocalContext.current
    SnyggBox(
        elementName = FlorisImeUi.ClipboardItem.elementName,
        attributes = mapOf("type" to "text"),
        modifier = Modifier
            .fillMaxWidth()
            .semantics { selected = option.selected },
        clickAndSemanticsModifier = Modifier.clickable(role = Role.Button, onClick = onClick),
    ) {
        SnyggRow(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            SnyggColumn(modifier = Modifier.weight(1f)) {
                SnyggText(text = context.getString(option.style.label))
                SnyggText(
                    elementName = FlorisImeUi.ClipboardItemTimestamp.elementName,
                    text = context.getString(option.style.hint),
                )
            }
            val (icon, tag) = when {
                !option.unlocked -> Icons.Outlined.Lock to R.string.panel_tag_locked
                !option.installed -> Icons.Default.Download to R.string.panel_tag_get
                option.selected -> Icons.Default.Check to R.string.panel_tag_selected
                option.free -> null to R.string.panel_tag_free
                else -> null to null
            }
            if (icon != null) SnyggIcon(modifier = Modifier.padding(start = 8.dp).size(18.dp), imageVector = icon)
            if (tag != null) SnyggText(modifier = Modifier.padding(start = 6.dp), text = stringResource(tag))
        }
    }
}
