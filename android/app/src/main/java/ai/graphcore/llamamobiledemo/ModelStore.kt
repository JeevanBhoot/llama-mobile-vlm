// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

package ai.graphcore.llamamobiledemo

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import java.io.File

sealed interface ModelDownloadState {
    data object NotDownloadable : ModelDownloadState
    data object Missing : ModelDownloadState
    data object Installed : ModelDownloadState
    data class Downloading(val progress: Double?) : ModelDownloadState
    data class Failed(val message: String) : ModelDownloadState
}

class ModelStore(context: Context) {
    private val appContext = context.applicationContext
    private val downloadManager =
        appContext.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    private val prefs = appContext.getSharedPreferences("model-downloads", Context.MODE_PRIVATE)
    private val rootDir = appContext.getExternalFilesDir(null) ?: appContext.filesDir

    fun pathFor(model: Model): String {
        return model.fileName?.let { modelFile(it).absolutePath } ?: ""
    }

    fun state(model: Model): ModelDownloadState {
        if (model.downloadUrl == null || model.fileName == null) {
            return ModelDownloadState.NotDownloadable
        }
        if (modelFile(model.fileName).exists()) {
            return ModelDownloadState.Installed
        }

        val downloadId = prefs.getLong(downloadIdKey(model), -1L)
        if (downloadId == -1L) {
            return ModelDownloadState.Missing
        }

        return queryDownload(model, downloadId)
    }

    fun startDownload(model: Model): ModelDownloadState {
        val url = model.downloadUrl ?: return ModelDownloadState.NotDownloadable
        val fileName = model.fileName ?: return ModelDownloadState.NotDownloadable
        val currentState = state(model)
        if (currentState is ModelDownloadState.Downloading || currentState is ModelDownloadState.Installed) {
            return currentState
        }

        modelFile(fileName).delete()
        tempModelFile(fileName).delete()

        val request = DownloadManager.Request(Uri.parse(url))
            .setTitle(model.label)
            .setDescription("Downloading model")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(false)
            .setDestinationInExternalFilesDir(appContext, null, "models/$fileName.download")

        val downloadId = downloadManager.enqueue(request)
        prefs.edit().putLong(downloadIdKey(model), downloadId).apply()
        return ModelDownloadState.Downloading(null)
    }

    fun delete(model: Model) {
        model.fileName?.let {
            modelFile(it).delete()
            tempModelFile(it).delete()
        }

        val downloadId = prefs.getLong(downloadIdKey(model), -1L)
        if (downloadId != -1L) {
            downloadManager.remove(downloadId)
            prefs.edit().remove(downloadIdKey(model)).apply()
        }
    }

    private fun queryDownload(model: Model, downloadId: Long): ModelDownloadState {
        val cursor = downloadManager.query(DownloadManager.Query().setFilterById(downloadId))
            ?: return clearDownload(model, ModelDownloadState.Missing)

        cursor.use {
            if (!it.moveToFirst()) {
                return clearDownload(model, ModelDownloadState.Missing)
            }

            return when (it.getIntColumn(DownloadManager.COLUMN_STATUS)) {
                DownloadManager.STATUS_PENDING,
                DownloadManager.STATUS_PAUSED,
                DownloadManager.STATUS_RUNNING -> {
                    val downloaded = it.getLongColumn(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
                    val total = it.getLongColumn(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
                    val progress = if (total > 0) downloaded.toDouble() / total else null
                    ModelDownloadState.Downloading(progress)
                }

                DownloadManager.STATUS_SUCCESSFUL -> finishDownload(model)
                DownloadManager.STATUS_FAILED -> {
                    val reason = it.getIntColumn(DownloadManager.COLUMN_REASON)
                    clearDownload(model, ModelDownloadState.Failed("Download failed: $reason"))
                }

                else -> clearDownload(model, ModelDownloadState.Missing)
            }
        }
    }

    private fun finishDownload(model: Model): ModelDownloadState {
        val fileName = model.fileName ?: return ModelDownloadState.NotDownloadable
        val tempFile = tempModelFile(fileName)
        val finalFile = modelFile(fileName)
        if (!tempFile.exists()) {
            return clearDownload(model, ModelDownloadState.Failed("Downloaded file was not found"))
        }
        finalFile.parentFile?.mkdirs()
        finalFile.delete()
        if (!tempFile.renameTo(finalFile)) {
            tempFile.delete()
            return clearDownload(model, ModelDownloadState.Failed("Could not finalize downloaded model"))
        }
        return clearDownload(model, ModelDownloadState.Installed)
    }

    private fun clearDownload(model: Model, state: ModelDownloadState): ModelDownloadState {
        prefs.edit().remove(downloadIdKey(model)).apply()
        return state
    }

    private fun downloadIdKey(model: Model) = "download-id-${model.name}"

    private fun modelFile(fileName: String) = File(rootDir, "models/$fileName")

    private fun tempModelFile(fileName: String) = File(rootDir, "models/$fileName.download")

    private fun android.database.Cursor.getIntColumn(name: String): Int {
        return getInt(getColumnIndexOrThrow(name))
    }

    private fun android.database.Cursor.getLongColumn(name: String): Long {
        return getLong(getColumnIndexOrThrow(name))
    }
}
