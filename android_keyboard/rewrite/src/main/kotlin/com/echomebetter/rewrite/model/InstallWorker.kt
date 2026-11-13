package com.echomebetter.rewrite.model

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.text.format.Formatter
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.echomebetter.rewrite.R
import com.echomebetter.rewrite.rewrite
import com.echomebetter.modelstore.InstallProgress
import com.echomebetter.modelstore.ModelStoreException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Downloads one [InstallTarget] as a foreground job with a progress notification. A network failure is retried by
 * WorkManager (the downloader resumes from what it has); anything else fails the job with a reason for the setup
 * screen to show.
 */
class InstallWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    private val notifications = context.getSystemService(NotificationManager::class.java)

    override suspend fun doWork(): Result {
        val target = InstallTarget.parse(inputData.getString(KEY_TARGET)) ?: return Result.failure()
        setForeground(foregroundInfo(null))
        val progress = MutableStateFlow<InstallProgress?>(null)
        return try {
            coroutineScope {
                // The download reports every chunk; the screen and the notification need a few updates a second.
                val reporter = launch {
                    var shown: InstallProgress? = null
                    while (true) {
                        delay(PROGRESS_INTERVAL_MS)
                        val current = progress.value ?: continue
                        if (current == shown) continue
                        shown = current
                        setProgress(workDataOf(KEY_DONE to current.doneBytes, KEY_TOTAL to current.totalBytes))
                        notifications.notify(NOTIFICATION_ID, notification(current))
                    }
                }
                applicationContext.rewrite.models.runInstall(target) { progress.value = it }
                reporter.cancel()
            }
            Result.success()
        } catch (error: CancellationException) {
            throw error
        } catch (error: ModelStoreException.Network) {
            if (runAttemptCount < MAX_ATTEMPTS) Result.retry() else failure(FailureReason.NETWORK)
        } catch (error: ModelStoreException.InsufficientStorage) {
            failure(FailureReason.STORAGE, error.neededBytes - error.freeBytes)
        } catch (error: ModelStoreException.Integrity) {
            failure(FailureReason.INTEGRITY)
        } catch (error: ModelStoreException.BaseChanged) {
            failure(FailureReason.BASE_CHANGED)
        } catch (error: ModelStoreException.NoSuchAdapter) {
            failure(FailureReason.NO_ADAPTER)
        } catch (error: ModelStoreException.BaseMissing) {
            failure(FailureReason.UNKNOWN)
        } catch (error: com.echomebetter.engine.manifest.ManifestException) {
            failure(FailureReason.INTEGRITY)
        } catch (error: java.io.IOException) {
            failure(FailureReason.UNKNOWN)
        }
    }

    override suspend fun getForegroundInfo(): ForegroundInfo = foregroundInfo(null)

    private fun failure(reason: FailureReason, neededBytes: Long = 0) =
        Result.failure(workDataOf(KEY_REASON to reason.name, KEY_NEEDED to neededBytes))

    private fun foregroundInfo(progress: InstallProgress?): ForegroundInfo =
        ForegroundInfo(NOTIFICATION_ID, notification(progress), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)

    private fun notification(progress: InstallProgress?): android.app.Notification {
        val context = applicationContext
        notifications.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, context.getString(R.string.notification_channel), NotificationManager.IMPORTANCE_LOW),
        )
        val builder = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(context.getString(R.string.notification_title))
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
        if (progress == null || progress.totalBytes == 0L) {
            builder.setProgress(0, 0, true)
        } else {
            builder.setProgress(PROGRESS_SCALE, (PROGRESS_SCALE * progress.doneBytes / progress.totalBytes).toInt(), false)
            builder.setContentText(
                context.getString(
                    R.string.notification_progress,
                    Formatter.formatShortFileSize(context, progress.doneBytes),
                    Formatter.formatShortFileSize(context, progress.totalBytes),
                ),
            )
        }
        return builder.build()
    }

    internal companion object {
        const val KEY_TARGET = "target"
        const val KEY_DONE = "done"
        const val KEY_TOTAL = "total"
        const val KEY_REASON = "reason"
        const val KEY_NEEDED = "needed"

        private const val CHANNEL_ID = "model-downloads"
        private const val NOTIFICATION_ID = 1
        private const val PROGRESS_INTERVAL_MS = 500L
        private const val PROGRESS_SCALE = 1000
        private const val MAX_ATTEMPTS = 5
    }
}
