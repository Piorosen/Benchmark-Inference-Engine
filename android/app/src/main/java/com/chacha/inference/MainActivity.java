package com.chacha.inference;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        OrtSession session;
    }
}