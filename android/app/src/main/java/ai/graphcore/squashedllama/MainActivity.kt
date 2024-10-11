package ai.graphcore.squashedllama

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@SuppressLint("DefaultLocale")
@Composable
fun MainScreen(
    ready: Boolean,
    output: String,
    promptForOutput: String,
    prefillRate: Double?,
    generationRate: Double?,
    modifier: Modifier = Modifier,
    onLoad: () -> Unit = {},
    onUnload: () -> Unit = {},
    onSubmitPrompt: (String) -> Unit = {}
) {
    Surface(modifier = modifier, color = MaterialTheme.colorScheme.surface) {
        Column(
            modifier = Modifier
                .padding(24.dp)
                .imePadding(),
            verticalArrangement = Arrangement.Bottom
        ) {
            var prompt by rememberSaveable { mutableStateOf("") }
            val focusManager = LocalFocusManager.current
            val textStyle = MaterialTheme.typography.bodyLarge

            Row(modifier = Modifier.align(Alignment.End)) {
                Button(onLoad) {
                    Text("Load")
                }
                TextButton(onUnload) {
                    Text("Unload")
                }
            }
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                enabled = ready,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = {
                    focusManager.clearFocus()
                    onSubmitPrompt(prompt)
                }),
                label = { Text("Prompt") },
                textStyle = textStyle,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(16.dp))
            OutlinedTextField(
                value = output,
                onValueChange = {},
                readOnly = true,
                enabled = ready,
                minLines = 3,
                label = { Text("Output") },
                textStyle = if (promptForOutput != prompt) textStyle.copy(color = Color.Gray) else textStyle,
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(16.dp))
            fun fmtRate(rate: Double?): String {
                return if (rate == null) "--" else "%.1f".format(rate)
            }
            Text(
                String.format(
                    "Prefill ${fmtRate(prefillRate)} tok/s | Generation ${fmtRate(generationRate)} tok/s"
                ),
                style = MaterialTheme.typography.labelSmall,
                color = Color.Gray,
            )
        }
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        var ready by mutableStateOf(false)
        var output by mutableStateOf("")
        var promptForOutput by mutableStateOf("")
        var prefillRate by mutableStateOf<Double?>(null)
        var generationRate by mutableStateOf<Double?>(null)

        Worker.setListener { event ->
            when (event) {
                is Worker.Event.Loaded -> {
                    ready = event.path != null
                    output = ""
                    promptForOutput = ""
                    prefillRate = null
                    generationRate = null
                }

                is Worker.Event.Response -> {
                    output = event.text
                    promptForOutput = event.prompt
                    prefillRate = event.prefillRate
                    generationRate = event.generationRate
                }
            }
        }

        val modelPath = "/data/local/tmp/Llama-3.2-1B-Instruct-BF16.sqt"

        setContent {
            CustomTheme {
                MainScreen(
                    ready = ready,
                    output = output,
                    promptForOutput = promptForOutput,
                    prefillRate = prefillRate,
                    generationRate = generationRate,
                    modifier = Modifier.fillMaxSize(),
                    onSubmitPrompt = { Worker.send(Worker.Command.Generate(it)) },
                    onLoad = { Worker.send(Worker.Command.Load(modelPath)) },
                    onUnload = { Worker.send(Worker.Command.Unload) },
                )
            }
        }
    }
}

@Preview(showSystemUi = true, uiMode = Configuration.UI_MODE_NIGHT_NO, name = "Preview Light")
@Preview(showSystemUi = true, uiMode = Configuration.UI_MODE_NIGHT_YES, name = "Preview Dark")
@Composable
fun Preview() {
    CustomTheme {
        MainScreen(
            ready = true,
            output = "That's a very interesting question. The answer is subjective.",
            promptForOutput = "",
            prefillRate = 0.5,
            generationRate = null,
            modifier = Modifier.fillMaxSize(),
        )
    }
}
