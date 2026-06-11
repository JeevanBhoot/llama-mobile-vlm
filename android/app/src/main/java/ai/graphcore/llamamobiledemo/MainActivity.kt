// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

package ai.graphcore.llamamobiledemo

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview as CameraPreviewUseCase
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image as ComposeImage
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

private const val MAX_IMAGE_PREVIEW_SIZE = 560

data class SelectedImage(
    val image: Image,
    val preview: Bitmap,
    val label: String
)

fun selectedImageFromUri(context: Context, uri: Uri): SelectedImage {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE)
        targetImageSize(
            width = info.size.width,
            height = info.size.height,
            maxSize = MAX_IMAGE_PREVIEW_SIZE
        )?.let { size ->
            decoder.setTargetSize(size.first, size.second)
        }
    }
    return selectedImageFromBitmap(bitmap, uri.lastPathSegment ?: "Selected image")
}

fun selectedImageFromBitmap(bitmap: Bitmap, label: String): SelectedImage {
    val scaledBitmap = resizeBitmapToFit(bitmap, MAX_IMAGE_PREVIEW_SIZE)
    val argbBitmap = compactArgb8888Bitmap(scaledBitmap)
    if (argbBitmap !== scaledBitmap) scaledBitmap.recycle()
    val expectedByteCount = argbBitmap.width * argbBitmap.height * 4
    require(argbBitmap.byteCount == expectedByteCount) {
        "Expected compact ARGB_8888 bitmap with $expectedByteCount bytes, got ${argbBitmap.byteCount}"
    }
    val argb = ByteBuffer
        .allocateDirect(expectedByteCount)
        .order(ByteOrder.nativeOrder())
    argbBitmap.copyPixelsToBuffer(argb.asIntBuffer())
    argb.rewind()
    return SelectedImage(
        image = Image(
            width = argbBitmap.width,
            height = argbBitmap.height,
            argb = argb
        ),
        preview = argbBitmap,
        label = label
    )
}

fun targetImageSize(width: Int, height: Int, maxSize: Int): Pair<Int, Int>? {
    val largestSide = maxOf(width, height)
    if (width <= 0 || height <= 0 || largestSide <= maxSize) return null

    val targetWidth = ((width.toLong() * maxSize) / largestSide).toInt().coerceAtLeast(1)
    val targetHeight = ((height.toLong() * maxSize) / largestSide).toInt().coerceAtLeast(1)
    return targetWidth to targetHeight
}

fun resizeBitmapToFit(bitmap: Bitmap, maxSize: Int): Bitmap {
    val largestSide = maxOf(bitmap.width, bitmap.height)
    if (largestSide <= maxSize) return bitmap

    val scale = maxSize.toFloat() / largestSide
    val width = (bitmap.width * scale).toInt().coerceAtLeast(1)
    val height = (bitmap.height * scale).toInt().coerceAtLeast(1)
    val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
    if (resized != bitmap) bitmap.recycle()
    return resized
}

fun compactArgb8888Bitmap(bitmap: Bitmap): Bitmap {
    if (bitmap.config == Bitmap.Config.ARGB_8888 && bitmap.byteCount == bitmap.width * bitmap.height * 4) {
        return bitmap
    }
    val argbBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
    Canvas(argbBitmap).drawBitmap(bitmap, 0f, 0f, null)
    return argbBitmap
}

@SuppressLint("DefaultLocale")
@Composable
fun MainScreen(
    ready: Boolean,
    selectedModel: Model,
    selectedImage: SelectedImage?,
    cameraActive: Boolean,
    output: String,
    outputIsError: Boolean,
    promptForOutput: String,
    loadingProgress: Double?,
    prefillProgress: Double?,
    prefillTime: Double?,
    generationRate: Double?,
    modifier: Modifier = Modifier,
    onModelSelected: (Model) -> Unit = {},
    onSelectImage: () -> Unit = {},
    onTakePhoto: () -> Unit = {},
    onCaptureCameraImage: (Bitmap) -> Unit = {},
    onClearImage: () -> Unit = {},
    onShowAbout: () -> Unit = {},
    onSubmitPrompt: (String) -> Unit = {}
) {
    Surface(modifier = modifier, color = MaterialTheme.colorScheme.surface) {
        Box(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp)
                    .imePadding(),
                verticalArrangement = Arrangement.Bottom
            ) {
                var prompt by rememberSaveable { mutableStateOf("") }
                val focusManager = LocalFocusManager.current
                val textStyle = MaterialTheme.typography.bodyLarge
                val submitPrompt = {
                    focusManager.clearFocus()
                    onSubmitPrompt(prompt)
                }

                ImageSelector(
                    enabled = selectedModel.supportsImage,
                    selectedImage = selectedImage.takeIf { selectedModel.supportsImage },
                    cameraActive = cameraActive && selectedModel.supportsImage,
                    onSelectImage = onSelectImage,
                    onTakePhoto = onTakePhoto,
                    onCaptureCameraImage = onCaptureCameraImage,
                    onClearImage = onClearImage,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(8.dp))
                ModelSelector(
                    selectedModel = selectedModel,
                    onModelSelected = onModelSelected,
                    modifier = Modifier.fillMaxWidth()
                )
                ProgressBar(
                    label = "Loading",
                    progress = loadingProgress,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = prompt,
                        onValueChange = { prompt = it },
                        enabled = ready,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        keyboardActions = KeyboardActions(onSend = { submitPrompt() }),
                        label = { Text("Prompt") },
                        trailingIcon = {
                            if (prompt.isNotEmpty()) {
                                IconButton(onClick = { prompt = "" }) {
                                    Icon(
                                        imageVector = Icons.Filled.Clear,
                                        contentDescription = "Clear prompt"
                                    )
                                }
                            }
                        },
                        textStyle = textStyle,
                        modifier = Modifier.weight(1f),
                    )
                    IconButton(
                        onClick = { submitPrompt() },
                        enabled = ready,
                        modifier = Modifier
                            .padding(start = 8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.Send,
                            contentDescription = "Send",
                            tint = if (ready) Color.Gray else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
                        )
                    }
                }
                ProgressBar(
                    label = "Prefill",
                    progress = prefillProgress,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(16.dp))
                OutlinedTextField(
                    value = output,
                    onValueChange = {},
                    readOnly = true,
                    enabled = ready,
                    minLines = 3,
                    label = { Text("Output") },
                    textStyle = when {
                        outputIsError -> textStyle.copy(color = MaterialTheme.colorScheme.error)
                        promptForOutput != prompt -> textStyle.copy(color = Color.Gray)
                        else -> textStyle
                    },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(16.dp))
                fun fmtTime(time: Double?): String {
                    return if (time == null) "--" else "%.1f s".format(time)
                }
                fun fmtRate(rate: Double?): String {
                    return if (rate == null) "--" else "%.1f tok/s".format(rate)
                }
                Text(
                    String.format(
                        "Prefill ${fmtTime(prefillTime)} | Generation ${fmtRate(generationRate)}"
                    ),
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.Gray,
                )
            }
            IconButton(
                onClick = onShowAbout,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .statusBarsPadding()
                    .padding(16.dp)
            ) {
                Icon(
                    imageVector = Icons.Filled.Settings,
                    contentDescription = "About",
                    tint = Color.Gray
                )
            }
        }
    }
}

@SuppressLint("DefaultLocale")
@Composable
fun ProgressBar(
    label: String,
    progress: Double?,
    modifier: Modifier = Modifier
) {
    if (progress == null) return

    Column(modifier = modifier.padding(top = 4.dp)) {
        LinearProgressIndicator(
            progress = { progress.toFloat() },
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(Modifier.height(2.dp))
        Text(
            text = "$label ${"%.0f".format(progress * 100)}%",
            style = MaterialTheme.typography.labelSmall,
            color = Color.Gray
        )
    }
}

@Composable
fun ImageSelector(
    enabled: Boolean,
    selectedImage: SelectedImage?,
    cameraActive: Boolean,
    onSelectImage: () -> Unit,
    onTakePhoto: () -> Unit,
    onCaptureCameraImage: (Bitmap) -> Unit,
    onClearImage: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier
            .aspectRatio(1f),
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surface,
        border = if (selectedImage == null) {
            BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
        } else {
            null
        }
    ) {
        Box(contentAlignment = Alignment.Center) {
            if (cameraActive && selectedImage == null) {
                CameraPreviewBox(
                    onCaptureCameraImage = onCaptureCameraImage,
                    onCancel = onClearImage,
                    modifier = Modifier.fillMaxSize()
                )
            } else if (selectedImage == null) {
                if (enabled) {
                    Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                        IconButton(
                            onClick = onSelectImage,
                            modifier = Modifier.size(72.dp)
                        ) {
                            Icon(
                                painter = painterResource(R.drawable.ic_photo_library_24),
                                contentDescription = "Choose image",
                                modifier = Modifier.size(36.dp)
                            )
                        }
                        IconButton(
                            onClick = onTakePhoto,
                            modifier = Modifier.size(72.dp)
                        ) {
                            Icon(
                                painter = painterResource(R.drawable.ic_photo_camera_24),
                                contentDescription = "Take photo",
                                modifier = Modifier.size(36.dp)
                            )
                        }
                    }
                } else {
                    Text(
                        "Model does not support images",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color.Gray
                    )
                }
            } else {
                val imageAspectRatio =
                    selectedImage.preview.width.toFloat() / selectedImage.preview.height.toFloat()
                Surface(
                    modifier = if (imageAspectRatio >= 1f) {
                        Modifier
                            .fillMaxWidth()
                            .aspectRatio(imageAspectRatio)
                    } else {
                        Modifier
                            .fillMaxHeight()
                            .aspectRatio(imageAspectRatio)
                    },
                    shape = MaterialTheme.shapes.medium,
                    color = MaterialTheme.colorScheme.surface,
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
                ) {
                    Box {
                        ComposeImage(
                            bitmap = selectedImage.preview.asImageBitmap(),
                            contentDescription = selectedImage.label,
                            contentScale = ContentScale.Fit,
                            modifier = Modifier.fillMaxSize()
                        )
                        Surface(
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(8.dp),
                            shape = MaterialTheme.shapes.small,
                            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f)
                        ) {
                            IconButton(onClick = onClearImage) {
                                Icon(
                                    imageVector = Icons.Filled.Clear,
                                    contentDescription = "Clear image"
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun CameraPreviewBox(
    onCaptureCameraImage: (Bitmap) -> Unit,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var capturing by remember { mutableStateOf(false) }
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
        }
    }

    DisposableEffect(context, lifecycleOwner, previewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val executor = ContextCompat.getMainExecutor(context)
        val listener = Runnable {
            val cameraProvider = cameraProviderFuture.get()
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(MAX_IMAGE_PREVIEW_SIZE, MAX_IMAGE_PREVIEW_SIZE),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                    )
                )
                .build()
            val preview = CameraPreviewUseCase.Builder()
                .setResolutionSelector(resolutionSelector)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview
            )
        }
        cameraProviderFuture.addListener(listener, executor)

        onDispose {
            if (cameraProviderFuture.isDone) {
                cameraProviderFuture.get().unbindAll()
            }
        }
    }

    Box(modifier = modifier) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier
                .fillMaxSize()
                .clickable {
                    val bitmap = previewView.bitmap
                    if (!capturing && bitmap != null) {
                        capturing = true
                        onCaptureCameraImage(bitmap)
                    }
                }
        )
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 24.dp)
                .size(72.dp)
                .clip(CircleShape)
                .background(Color.White.copy(alpha = if (capturing) 0.45f else 0.85f))
                .padding(6.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = if (capturing) 0.35f else 0.0f))
            )
        }
        Surface(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(8.dp),
            shape = MaterialTheme.shapes.small,
            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f)
        ) {
            IconButton(onClick = onCancel) {
                Icon(
                    imageVector = Icons.Filled.Clear,
                    contentDescription = "Close camera"
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelSelector(
    selectedModel: Model,
    onModelSelected: (Model) -> Unit,
    modifier: Modifier = Modifier
) {
    var expanded by rememberSaveable { mutableStateOf(false) }

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = modifier
    ) {
        OutlinedTextField(
            value = selectedModel.label,
            onValueChange = {},
            readOnly = true,
            label = { Text("Model") },
            trailingIcon = {
                ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
            },
            modifier = Modifier
                .menuAnchor()
                .fillMaxWidth()
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            Model.entries.forEach { model ->
                DropdownMenuItem(
                    text = { Text(model.label) },
                    onClick = {
                        expanded = false
                        onModelSelected(model)
                    }
                )
            }
        }
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        var selectedModel by mutableStateOf(Model.Dummy)
        var selectedImage by mutableStateOf<SelectedImage?>(null)
        var loadedModel by mutableStateOf<Model?>(Model.Dummy)
        var output by mutableStateOf("")
        var outputIsError by mutableStateOf(false)
        var promptForOutput by mutableStateOf("")
        var loadingProgress by mutableStateOf<Double?>(null)
        var prefillProgress by mutableStateOf<Double?>(null)
        var prefillTime by mutableStateOf<Double?>(null)
        var generationRate by mutableStateOf<Double?>(null)
        var cameraActive by mutableStateOf(false)
        var showAbout by mutableStateOf(false)

        Worker.setListener { event ->
            when (event) {
                is Worker.Event.Loaded -> {
                    loadedModel = event.model
                    output = ""
                    outputIsError = false
                    promptForOutput = ""
                    loadingProgress = null
                    prefillProgress = null
                    prefillTime = null
                    generationRate = null
                }

                is Worker.Event.Progress -> {
                    when (event.phase) {
                        ProgressPhase.Loading -> loadingProgress = event.progress
                        ProgressPhase.Prefill -> prefillProgress = event.progress
                    }
                }

                is Worker.Event.Response -> {
                    output = event.text
                    outputIsError = false
                    promptForOutput = event.prompt
                    prefillProgress = null
                    prefillTime = event.prefillTime
                    generationRate = event.generationRate
                }

                is Worker.Event.Error -> {
                    output = event.message
                    outputIsError = true
                    promptForOutput = ""
                    loadingProgress = null
                    prefillProgress = null
                    prefillTime = null
                    generationRate = null
                }
            }
        }

        setContent {
            val context = LocalContext.current
            fun loadSelectedImage(uri: Uri) {
                thread {
                    val image = selectedImageFromUri(context.applicationContext, uri)
                    runOnUiThread {
                        selectedImage = image
                    }
                }
            }

            val imagePicker = rememberLauncherForActivityResult(
                ActivityResultContracts.PickVisualMedia()
            ) { uri ->
                if (uri != null) {
                    loadSelectedImage(uri)
                }
            }
            val cameraPermission = rememberLauncherForActivityResult(
                ActivityResultContracts.RequestPermission()
            ) { granted ->
                if (granted) {
                    cameraActive = true
                }
            }

            CustomTheme(largeFonts = false) {
                if (showAbout) {
                    AboutScreen(
                        onBack = { showAbout = false },
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    MainScreen(
                        ready = (loadedModel == selectedModel),
                        selectedModel = selectedModel,
                        selectedImage = selectedImage,
                        cameraActive = cameraActive,
                        output = output,
                        outputIsError = outputIsError,
                        promptForOutput = promptForOutput,
                        loadingProgress = loadingProgress,
                        prefillProgress = prefillProgress,
                        prefillTime = prefillTime,
                        generationRate = generationRate,
                        modifier = Modifier.fillMaxSize(),
                        onModelSelected = { model: Model ->
                            cameraActive = false
                            selectedModel = model
                            Worker.send(Worker.Command.Load(model))
                        },
                        onSelectImage = {
                            imagePicker.launch(
                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                            )
                        },
                        onTakePhoto = {
                            if (
                                ContextCompat.checkSelfPermission(
                                    context,
                                    Manifest.permission.CAMERA
                                ) == PackageManager.PERMISSION_GRANTED
                            ) {
                                selectedImage = null
                                cameraActive = true
                            } else {
                                cameraPermission.launch(Manifest.permission.CAMERA)
                            }
                        },
                        onCaptureCameraImage = { bitmap ->
                            thread {
                                val image = selectedImageFromBitmap(bitmap, "Camera image")
                                runOnUiThread {
                                    selectedImage = image
                                    cameraActive = false
                                }
                            }
                        },
                        onClearImage = {
                            selectedImage = null
                            cameraActive = false
                        },
                        onShowAbout = {
                            cameraActive = false
                            showAbout = true
                        },
                        onSubmitPrompt = { prompt ->
                            val image = selectedImage.takeIf { selectedModel.supportsImage }
                            Worker.send(
                                Worker.Command.Generate(
                                    prompt = prompt,
                                    image = image?.image
                                )
                            )
                        },
                    )
                }
            }
        }
    }
}

@Preview(showSystemUi = true, uiMode = Configuration.UI_MODE_NIGHT_NO, name = "Preview Light")
//@Preview(showSystemUi = true, uiMode = Configuration.UI_MODE_NIGHT_YES, name = "Preview Dark")
@Composable
fun Preview() {
    CustomTheme {
        MainScreen(
            ready = true,
            selectedModel = Model.Dummy,
            selectedImage = null,
            cameraActive = false,
            output = "That's a very interesting question. The answer is subjective.",
            outputIsError = false,
            promptForOutput = "",
            loadingProgress = null,
            prefillProgress = null,
            prefillTime = 0.5,
            generationRate = null,
            modifier = Modifier.fillMaxSize(),
        )
    }
}
