#include <array>
#include <cmath>
#include <iostream>
using namespace std;


double square (double x) {
  double squared = x * x;
  return squared;
}

struct standardizingData
{
  double mean;
  double standardDeviation;
};

standardizingData standardize (array<double, 5>& v) {
  standardizingData d;
  double sum = 0;
  for (double x : v) sum += x;
  d.mean = sum / v.size();
  double deviationSum = 0;
  for (double& x : v) {
    x -= d.mean;
    deviationSum += square(x);
  }
  double variance = deviationSum / v.size();
  d.standardDeviation = sqrt(variance);
  for (double& x : v) x /= d.standardDeviation;
  return d;
}

double standardizeSingle (double x, standardizingData d) {
  x -= d.mean;
  x /= d.standardDeviation;
  return x;
}

double unStandardize (double x, standardizingData d) {
  x *= d.standardDeviation;
  x += d.mean;
  return x;
}

double computeWeightSlopeMSE (double error, double isolatedParameter) {
  return 2 * error * isolatedParameter;
}

double computeBiasSlopeMSE (double error) {
  return 2 * error;
}

double computeWeightGradientMAE (double error, double isolatedParameter) {
  double sign = (error > 0) - (error < 0);
  return isolatedParameter * sign;
}

double computeBiasGradientMAE (double error) {
  double sign = (error > 0) - (error < 0);
  return sign;
}

double checkGradient(double w1, double w2, double b) {
    return 0;
}

int main() {
  int epochs = 1000;
  double checkGradientStep = 0.0001;
  double learningRate = 0.01;
  double w1 = 0, w2 = 0, b = 0;
  array<double, 5> x1 = {2.5, 4.6, 1.9, 8.4, 9.1}; // uafhængige variabler / forklarende variabler
  array<double, 5> x2 = {3.2, 6.8, 1.5, 5.1, 2.7};
  array<double, 5> y = {6.3, 12.4, 4.5, 20.4, 21.8};
  standardizingData x1Standardization = standardize(x1);
  standardizingData x2Standardization = standardize(x2);
  standardizingData yStandardization = standardize(y);
  double w1Gradient, w2Gradient, bGradient;
  for (int epoch = 0; epoch < epochs; epoch++) {
    w1Gradient = 0, w2Gradient = 0, bGradient = 0;
    double loss = 0;
    double w1CheckGradient = 0, w2CheckGradient = 0, bCheckGradient = 0;
    for (size_t i = 0; i < x1.size(); i++) {
        double error = x1[i] * w1 + x2[i] * w2 +  b - y[i]; // fejl = prædikteret værdi - målvariabel / afhængige variabel
        w1Gradient += computeWeightSlopeMSE(error, x1[i]);
        w2Gradient += computeWeightSlopeMSE(error, x2[i]);
        bGradient += computeBiasSlopeMSE(error);
        loss += square(error);
        double w1PositiveCheck = square(x1[i] * (w1 + checkGradientStep) + x2[i] * w2 + b - y[i]);
        double w1NegativeCheck = square(x1[i] * (w1 - checkGradientStep) + x2[i] * w2 + b - y[i]);
        double w2PositiveCheck = square(x1[i] * w1 + x2[i] * (w2 + checkGradientStep) + b - y[i]);
        double w2NegativeCheck = square(x1[i] * w1 + x2[i] * (w2 - checkGradientStep) + b - y[i]);
        double bPositiveCheck = square(x1[i] * w1 + x2[i] * w2 + (b + checkGradientStep) - y[i]);
        double bNegativeCheck = square(x1[i] * w1 + x2[i] * w2 + (b - checkGradientStep) - y[i]);
        w1CheckGradient += (w1PositiveCheck - w1NegativeCheck) / (checkGradientStep * 2);
        w2CheckGradient += (w2PositiveCheck - w2NegativeCheck) / (checkGradientStep * 2);
        bCheckGradient += (bPositiveCheck - bNegativeCheck) / (checkGradientStep * 2);
    }
    size_t sample_size = x1.size();
    loss /= sample_size;
    w1Gradient /= sample_size;
    w2Gradient /= sample_size;
    bGradient /= sample_size;
    w1CheckGradient /= sample_size;
    w2CheckGradient /= sample_size;
    bCheckGradient /= sample_size;
    double w1Step = w1Gradient * learningRate;
    double w2Step = w2Gradient * learningRate;
    double bStep = bGradient * learningRate;
    w1 -= w1Step;
    w2 -= w2Step;
    b -= bStep;
    cout << "Calculated gradient: " << w1CheckGradient << ". Differentiated gradient: " << w1Gradient << ", ";
    cout << "Epoch: " << epoch << ", ";
    cout << "Loss: " << loss << ", ";
    cout << "W1: " << w1 << ", ";
    cout << "W2: " << w2 << ", ";
    cout << "B: " << b << "\n";
  }
  double input1 , input2;
  cout << "First feature sample to calculate: ";
  cin >> input1;
  cout << "Second feature sample to calculate: ";
  cin >> input2;

  input1 = standardizeSingle(input1, x1Standardization);
  input2 = standardizeSingle(input2, x2Standardization);

  double prediction = input1 * w1 + input2 * w2 + b;

  prediction = unStandardize(prediction, yStandardization);

  cout << "Prediction: " << prediction;
  return 0;
}
