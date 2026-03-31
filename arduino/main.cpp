#include <avr/pgmspace.h>

// Paste Pytorch model results
const float W1[48] PROGMEM = {0.316682, 0.090083, -2.719522, 1.155601, 1.349847, 2.009171, -2.560645, -1.190861, 0.935880, -2.591205, 0.692326, -1.199651, -1.813017, 1.463723, 1.247077, 0.703567, -1.477160, 1.988645, -2.423985, 0.588013, -1.504487, 1.322417, 2.165888, 0.718938, -2.420835, 2.279016, -0.012773, -1.105255, -1.586061, -1.239816, -0.828946, -1.886254, 1.493256, 0.679022, 1.235318, 2.207920, -3.014838, -0.810772, -0.115071, -0.779152, 1.499168, -2.132627, 2.421216, -0.920918, 1.179013, 2.142141, -2.139292, -0.458111};
const float b1[16] PROGMEM = {1.303843, -2.307303, 1.795186, 1.830976, -0.326688, -0.801306, 1.987897, -2.196120, 0.567898, 2.021583, 0.569053, -2.019458, 2.390813, 0.741497, -1.659449, -0.140696};
const float W2[128] PROGMEM = {-0.422026, 0.754507, 0.173652, -0.411239, -0.065777, -0.374985, -0.135794, -0.148345, 0.076455, 0.252115, -0.387888, 0.395016, -0.375000, 0.108278, 0.316037, -0.490490, 0.498545, -0.485760, -0.614226, -0.230028, -0.039886, 0.060741, 0.162201, 0.245677, -0.390019, -0.273666, 0.248964, -0.294780, -0.210118, 0.031942, 0.051411, -0.065321, -0.529483, 0.540693, -0.091476, -0.253504, 0.071087, 0.068671, -0.248652, 0.166054, 0.360121, -0.141713, -0.206047, 0.296206, -0.123026, 0.104792, 0.107800, -0.050367, 0.364473, 0.137363, 0.225491, 0.615174, 0.097217, 0.213709, 0.574488, 0.110948, 0.108531, 0.221897, -0.173483, -0.071567, 0.195941, -0.135185, -0.449140, -0.246858, -0.225328, -0.112388, 0.621271, -0.110705, -0.069274, -0.194126, -0.018135, -0.372190, 0.411464, -0.007576, -0.010089, -0.051895, 0.638909, -0.016038, -0.078543, 0.107457, 0.480854, -0.443766, -0.229970, -0.274731, -0.183350, 0.277441, -0.173886, -0.175496, -0.007686, 0.065831, -0.289082, -0.223255, -0.520605, 0.148066, 0.056514, 0.388461, -0.316882, 0.321323, -0.211293, -0.160741, -0.084728, 0.292494, -0.019562, 0.176988, 0.010966, -0.135455, 0.114771, 0.068185, -0.354235, -0.206765, 0.134820, 0.089135, -0.027704, -0.101649, -0.309649, 0.370169, -0.163523, 0.414435, -0.240442, 0.558586, 0.099916, 0.118237, -0.292063, 0.785777, -0.088241, -0.744229, -0.132662, -0.205344};
const float b2[8] PROGMEM = {0.277962, 0.748474, 0.111299, -0.664866, -0.170004, 0.842411, 0.240240, -0.051034};
const float W3[24] PROGMEM = {-0.144112, 0.345601, -0.259457, -0.177202, -0.231249, 0.201726, -0.028957, -0.370599, -0.466237, -0.294969, -0.221322, 0.165743, 0.320720, -0.356503, -0.331058, -0.374220, 0.455350, -0.045499, 0.408404, -0.513435, -0.423638, -0.217052, 0.324312, 0.411880};
const float b3[3] PROGMEM = {-0.061014, -0.048685, -0.480649};


const int ledPins[3] = {8, 9, 10}; 
const int NUM_COLORS = 4;

float test_colors[NUM_COLORS][3] = {
  {39.0 / 255.0, 245.0 / 255.0, 169.0 / 255.0}, // must be Green
  {122.0 / 255.0, 21.0 / 255.0, 230.0 / 255.0}, // must be Blue
  {177.0 / 255.0, 32.0 / 255.0, 106.0 / 255.0}, // must be Red
  {229.0 / 255.0, 232.0 / 255.0, 77.0 / 255.0} // must be Blue
};

void dense_layer(const float* input, const float* weights, const float* biases, 
                 float* output, int in_dim, int out_dim, bool use_relu) {
                     
  for (int i = 0; i < out_dim; i++) {
    float sum = pgm_read_float(&biases[i]); 
    for (int j = 0; j < in_dim; j++) {
      float w = pgm_read_float(&weights[i * in_dim + j]);
      sum += input[j] * w;
    }
    if (use_relu && sum < 0.0) {
      sum = 0.0;
    }
    output[i] = sum;
  }
}

int argmax(float* array, int size) {
  int max_idx = 0;
  float max_val = array[0];
  for (int i = 1; i < size; i++) {
    if (array[i] > max_val) {
      max_val = array[i];
      max_idx = i;
    }
  }
  return max_idx;
}
void setup() {
  Serial.begin(9600);
  
  for(int i = 0; i < 3; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW); 
  }

  while (!Serial); 
  Serial.println("Starting Multi-Color Sequence...");
  Serial.println("-------------------------");
}

void loop() {
  float hidden1[16];
  float hidden2[8];
  float output[3];

  for (int t = 0; t < NUM_COLORS; t++) {
    
    float* current_color = test_colors[t];

    Serial.print("Testing RGB: [");
    Serial.print(current_color[0] * 255.0, 0); Serial.print(", ");
    Serial.print(current_color[1] * 255.0, 0); Serial.print(", ");
    Serial.print(current_color[2] * 255.0, 0); Serial.println("]");

    dense_layer(current_color, W1, b1, hidden1, 3, 16, true);
    dense_layer(hidden1, W2, b2, hidden2, 16, 8, true);
    dense_layer(hidden2, W3, b3, output, 8, 3, false);

    int predicted_class = argmax(output, 3);

    for(int i = 0; i < 3; i++) {
      digitalWrite(ledPins[i], LOW);
    }
    
    digitalWrite(ledPins[predicted_class], HIGH);

    Serial.print("Predicted Label: ");
    Serial.println(predicted_class);
    Serial.println("-------------------------");

    delay(3000); 
  }

  Serial.println("\nRestarting Cycle...\n");
  for(int i = 0; i < 3; i++) {
    digitalWrite(ledPins[i], LOW);
  }
  delay(1000);
}