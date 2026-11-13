package com.echomebetter.keyboard.settings

import android.app.ActivityManager
import android.content.Context
import android.net.ConnectivityManager
import android.text.format.Formatter
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.echomebetter.engine.Style
import com.echomebetter.rewrite.R
import com.echomebetter.rewrite.RewriteContainer
import com.echomebetter.rewrite.access.StyleAccess
import com.echomebetter.rewrite.hint
import com.echomebetter.rewrite.label
import com.echomebetter.rewrite.model.DownloadState
import com.echomebetter.rewrite.model.FailureReason
import com.echomebetter.rewrite.model.InstallTarget
import com.echomebetter.rewrite.model.InstalledState
import com.echomebetter.rewrite.rewrite
import dev.patrickgold.florisboard.lib.compose.FlorisScreen
import dev.patrickgold.florisboard.lib.util.launchUrl
import kotlinx.coroutines.launch
import kotlin.math.ceil
import kotlin.math.roundToLong

/**
 * Settings → Rewrite: download the writing model (with the license agreement), add or remove styles, see the storage
 * used. The preview field at the bottom of the screen is FlorisBoard's own, for trying the keyboard right here.
 */
@Composable
fun RewriteSettingsScreen() = FlorisScreen {
    title = stringResource(R.string.settings_title)
    previewFieldVisible = true

    content {
        val context = LocalContext.current
        val feature = context.rewrite
        val scope = rememberCoroutineScope()
        val installed by feature.models.installed.collectAsState()
        val downloads by feature.models.downloads.collectAsState(emptyMap())
        val catalog by feature.models.catalog.collectAsState()
        val licenseAccepted by feature.settings.licenseAccepted.collectAsState()
        var pending by remember { mutableStateOf<List<InstallTarget>?>(null) }
        var confirmRemoval by remember { mutableStateOf(false) }

        LaunchedEffect(Unit) { feature.models.refreshCatalog() }

        val model = (installed as? InstalledState.Ready)?.model
        val firstDownload = listOf(InstallTarget.Base) + StyleAccess.FREE.map { InstallTarget.Adapter(it) }

        /** Wi-Fi: start now. Mobile data: ask first (the dialog below). */
        fun download(targets: List<InstallTarget>) {
            if (context.isOnMeteredNetwork()) pending = targets else feature.models.install(targets, allowMetered = false)
        }

        Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text(stringResource(R.string.settings_intro), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)

            ModelCard(
                feature = feature,
                ready = model != null,
                sizeBytes = catalog?.let { index -> index.base.sizeBytes + StyleAccess.FREE.sumOf { index.adapters[it]?.sizeBytes ?: 0 } },
                download = downloads[InstallTarget.Base],
                licenseAccepted = licenseAccepted,
                onLicense = feature.settings::setLicenseAccepted,
                onDownload = { download(firstDownload) },
                onCancel = feature.models::cancelDownloads,
            )

            SectionTitle(R.string.settings_styles_title)
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
                Style.entries.forEachIndexed { index, style ->
                    if (index > 0) HorizontalDivider(Modifier.padding(horizontal = 16.dp), color = MaterialTheme.colorScheme.outlineVariant)
                    StyleRow(
                        style = style,
                        access = feature.access,
                        modelReady = model != null,
                        installed = model?.manifest?.adapters?.containsKey(style) == true,
                        sizeBytes = catalog?.adapters?.get(style)?.sizeBytes,
                        download = downloads[InstallTarget.Adapter(style)],
                        onDownload = { download(listOf(InstallTarget.Adapter(style))) },
                        onRemove = { scope.launch { feature.models.removeAdapter(style) } },
                        onCancel = feature.models::cancelDownloads,
                    )
                }
            }
            if (!feature.access.isUnlocked(Style.PROFESSIONAL)) {
                UnlockCard()
            } else if (feature.config.unlockAllStyles) {
                Text(stringResource(R.string.settings_unlocked_debug), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            SectionTitle(R.string.settings_storage_title)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (model != null) stringResource(R.string.settings_storage_used, context.formatSize(model.sizeBytes)) else stringResource(R.string.settings_storage_none),
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.weight(1f),
                )
                if (model != null || downloads.isNotEmpty()) {
                    TextButton(onClick = { confirmRemoval = true }) {
                        Text(stringResource(R.string.settings_storage_remove_all), color = MaterialTheme.colorScheme.error)
                    }
                }
            }
            Text(stringResource(R.string.settings_privacy), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        pending?.let { targets ->
            val bytes = catalog?.let { index ->
                targets.sumOf { target ->
                    when (target) {
                        InstallTarget.Base -> index.base.sizeBytes
                        is InstallTarget.Adapter -> index.adapters[target.style]?.sizeBytes ?: 0
                    }
                }
            }
            AlertDialog(
                onDismissRequest = { pending = null },
                icon = { Icon(Icons.Default.Download, contentDescription = null) },
                title = { Text(stringResource(R.string.settings_metered_title)) },
                text = { Text(stringResource(R.string.settings_metered_body, bytes?.let(context::formatSize) ?: "~0.9 GB")) },
                confirmButton = {
                    TextButton(onClick = {
                        feature.models.install(targets, allowMetered = true)
                        pending = null
                    }) { Text(stringResource(R.string.settings_metered_use_data)) }
                },
                dismissButton = {
                    TextButton(onClick = {
                        feature.models.install(targets, allowMetered = false)
                        pending = null
                    }) { Text(stringResource(R.string.settings_metered_wait)) }
                },
            )
        }
        if (confirmRemoval) {
            AlertDialog(
                onDismissRequest = { confirmRemoval = false },
                title = { Text(stringResource(R.string.settings_storage_confirm_title)) },
                text = { Text(stringResource(R.string.settings_storage_confirm_body)) },
                confirmButton = {
                    TextButton(onClick = {
                        confirmRemoval = false
                        scope.launch {
                            feature.engine.release()
                            feature.models.removeAll()
                        }
                    }) { Text(stringResource(R.string.settings_storage_confirm), color = MaterialTheme.colorScheme.error) }
                },
                dismissButton = { TextButton(onClick = { confirmRemoval = false }) { Text(stringResource(R.string.settings_keep)) } },
            )
        }
    }
}

@Composable
private fun ModelCard(
    feature: RewriteContainer,
    ready: Boolean,
    sizeBytes: Long?,
    download: DownloadState?,
    licenseAccepted: Boolean,
    onLicense: (Boolean) -> Unit,
    onDownload: () -> Unit,
    onCancel: () -> Unit,
) {
    val context = LocalContext.current
    val memory = remember { context.deviceMemoryBytes() }
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    if (ready) Icons.Default.CheckCircle else Icons.Default.AutoAwesome,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                )
                Spacer(Modifier.width(10.dp))
                Text(stringResource(R.string.settings_model_title), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            }
            when {
                ready -> Text(stringResource(R.string.settings_model_ready), style = MaterialTheme.typography.bodyMedium)
                memory < feature.config.minDeviceMemoryBytes -> Text(
                    stringResource(R.string.settings_memory_too_low, gigabytes(memory, roundUp = false), gigabytes(feature.config.minDeviceMemoryBytes, roundUp = true)),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                )
                download != null -> {
                    Body(sizeBytes)
                    DownloadProgress(download, onCancel = onCancel, onRetry = onDownload)
                }
                else -> {
                    Body(sizeBytes)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Checkbox(checked = licenseAccepted, onCheckedChange = onLicense)
                        Text(stringResource(R.string.settings_license_consent), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        Button(onClick = onDownload, enabled = licenseAccepted) {
                            Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(6.dp))
                            Text(stringResource(R.string.settings_model_download))
                        }
                        TextButton(onClick = { context.launchUrl(LICENSE_URL) }) { Text(stringResource(R.string.settings_license_read)) }
                    }
                }
            }
        }
    }
}

@Composable
private fun Body(sizeBytes: Long?) {
    val context = LocalContext.current
    Text(
        if (sizeBytes != null) stringResource(R.string.settings_model_body, context.formatSize(sizeBytes)) else stringResource(R.string.settings_model_body_unknown),
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun DownloadProgress(download: DownloadState, onCancel: () -> Unit, onRetry: () -> Unit) {
    val context = LocalContext.current
    when (download) {
        is DownloadState.Queued -> {
            LinearProgressIndicator(Modifier.fillMaxWidth())
            ProgressLine(stringResource(if (download.waitingForWifi) R.string.settings_waiting_wifi else R.string.settings_queued), onCancel)
        }
        is DownloadState.Running -> {
            val fraction = if (download.totalBytes > 0) download.doneBytes.toFloat() / download.totalBytes else 0f
            LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
            ProgressLine(stringResource(R.string.settings_downloading, context.formatSize(download.doneBytes), context.formatSize(download.totalBytes)), onCancel)
        }
        is DownloadState.Failed -> {
            Text(failureText(context, download), color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
            OutlinedButton(onClick = onRetry) { Text(stringResource(R.string.settings_retry)) }
        }
    }
}

@Composable
private fun ProgressLine(text: String, onCancel: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(text, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
        TextButton(onClick = onCancel) { Text(stringResource(R.string.settings_cancel)) }
    }
}

@Composable
private fun StyleRow(
    style: Style,
    access: StyleAccess,
    modelReady: Boolean,
    installed: Boolean,
    sizeBytes: Long?,
    download: DownloadState?,
    onDownload: () -> Unit,
    onRemove: () -> Unit,
    onCancel: () -> Unit,
) {
    val context = LocalContext.current
    Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(stringResource(style.label), style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text(stringResource(style.hint), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (download != null && !access.isFree(style)) DownloadProgress(download, onCancel = onCancel, onRetry = onDownload)
        }
        Spacer(Modifier.width(8.dp))
        when {
            !access.isUnlocked(style) -> Icon(Icons.Outlined.Lock, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
            installed && access.isFree(style) -> Text(stringResource(R.string.settings_styles_included), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            access.isFree(style) -> Text(stringResource(R.string.settings_styles_free), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            installed -> TextButton(onClick = onRemove) { Text(stringResource(R.string.settings_styles_remove)) }
            !modelReady || download != null -> Unit
            else -> OutlinedButton(onClick = onDownload) {
                Text(sizeBytes?.let { stringResource(R.string.settings_styles_download, context.formatSize(it)) } ?: stringResource(R.string.settings_model_download))
            }
        }
    }
}

@Composable
private fun UnlockCard() {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.Lock, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.settings_unlock_title), style = MaterialTheme.typography.titleMedium)
            }
            Text(stringResource(R.string.settings_unlock_body), style = MaterialTheme.typography.bodyMedium)
            Button(onClick = {}, enabled = false) { Text(stringResource(R.string.settings_unlock_soon)) }
        }
    }
}

@Composable
private fun SectionTitle(title: Int) {
    Text(
        stringResource(title),
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(top = 4.dp).semantics { heading() },
    )
}

private const val LICENSE_URL = "https://huggingface.co/Thelivu/Thelivu-750M/blob/main/LICENSE"

private fun Context.formatSize(bytes: Long): String = Formatter.formatShortFileSize(this, bytes)

private fun Context.isOnMeteredNetwork(): Boolean = getSystemService(ConnectivityManager::class.java).isActiveNetworkMetered

private fun Context.deviceMemoryBytes(): Long =
    ActivityManager.MemoryInfo().also(getSystemService(ActivityManager::class.java)::getMemoryInfo).totalMem

/** Memory as phones are sold: whole gigabytes. */
private fun gigabytes(bytes: Long, roundUp: Boolean): String {
    val gb = bytes / 1e9
    return "${if (roundUp) ceil(gb).toLong() else gb.roundToLong()} GB"
}

private fun failureText(context: Context, failed: DownloadState.Failed): String = when (failed.reason) {
    FailureReason.NETWORK -> context.getString(R.string.settings_error_network)
    FailureReason.INTEGRITY -> context.getString(R.string.settings_error_integrity)
    FailureReason.STORAGE -> context.getString(R.string.settings_error_storage, context.formatSize(failed.neededBytes))
    FailureReason.BASE_CHANGED -> context.getString(R.string.settings_error_base_changed)
    FailureReason.NO_ADAPTER -> context.getString(R.string.settings_error_no_adapter)
    FailureReason.UNKNOWN -> context.getString(R.string.settings_error_unknown)
}
