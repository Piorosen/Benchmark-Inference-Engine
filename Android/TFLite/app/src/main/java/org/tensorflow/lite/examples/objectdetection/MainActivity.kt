/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.FloatBuffer


/**
 * Main entry point into our app. This app follows the single-activity pattern, and all
 * functionality is implemented in the form of fragments.
 */
class MainActivity : AppCompatActivity() {
    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
//        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(R.layout.activity_main)
        var button = findViewById<Button>(R.id.button);
        button.setOnClickListener {
            val models = arrayOf("alexnet", "googlenet", "mobilenet_v2", "resnet18", "resnet50", "resnet101", "vgg16")
            Toast.makeText(applicationContext, "hello world!", Toast.LENGTH_LONG).show()
            for (model in models){
                val options = Interpreter.Options()
                options.setUseXNNPACK(true)

                options.numThreads = 4
                val fileName = model + ".tflite"
                Log.i("CHACHA", "FILENAME : " + fileName)

                val modelFile = FileUtil.loadMappedFile(applicationContext, fileName)
                interpreter = Interpreter(modelFile, options)
                interpreter?.allocateTensors()

                val input_name = interpreter!!.getInputTensor(0).name()
                Log.i("CHACHA", input_name)

                for (i: Int in 1..100) {
                    val input = FloatBuffer.allocate(interpreter!!.getInputTensor(0).numElements())
                    val output = FloatBuffer.allocate(interpreter!!.getOutputTensor(0).numElements())

                    var inferenceTime = SystemClock.elapsedRealtimeNanos()
                    val result = interpreter?.run(input, output)
                    var endInference = SystemClock.elapsedRealtimeNanos()
                    Log.i("CHACHA", ((endInference-inferenceTime) / 1000.0 / 1000.0).toString())
                }

                interpreter!!.close()
            }
        }
    }

    override fun onBackPressed() {
    }

//
//    fun onResults(results: List<Detection>?, inferenceTime: Long, height: Int, width: Int) {
//        TODO("Not yet implemented")
//        Log.i("CHACHA", "RESULT!!")
//    }
}
