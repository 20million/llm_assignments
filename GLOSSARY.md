# 📖 AI Glossary for Beginners

A simple dictionary for the confusing words we use in this course.

---

## **F**

### **Feature** ($x$)
**What it is:** An input variable. A piece of information given to the AI.
*   **In Traditional ML:** "Height", "Weight", "Age".
*   **In LLMs:** A "Token" (part of a word).
*   **Analogy:** The ingredients for a recipe.

---

## **W**

### **Weight** ($w$ or $\theta$)
**What it is:** A knob the computer can turn. It decides how important a Feature is.
*   **High Weight:** This feature matters a lot.
*   **Zero Weight:** Ignore this feature.
*   **Analogy:** The volume dial on a radio. The AI learns the perfect volume settings.

---

## **B**

### **Bias (Offset)** ($b$)
**What it is:** An extra knob that shifts the result. It allows the model to output a non-zero value even if all inputs are zero.
*   **Math:** $y = mx + c$ (Here, $c$ is the Bias).
*   **Analogy:** If you are grading on a curve, everyone gets +5 points for free. That +5 is the Bias.

### **Bias (Statistical)**
**What it is:** An error caused by the model being too simple.
*   **Example:** Trying to fit a curve with a straight ruler.
*   **Opposite:** See *Variance*.

---

## **G**

### **Gradient** ($\nabla$)
**What it is:** The slope or "steepness" of the error.
*   **Purpose:** It points uphill. We want to go downhill (minimize error), so we go *opposite* the gradient.
*   **Analogy:** Feeling the ground with your foot to find which way is "down".

---

## **L**

### **Loss Function** ($J$)
**What it is:** The scoreboard. It calculates a single number representing "How bad is the AI right now?"
*   **Goal:** Make this number Zero.
*   **Examples:** Mean Squared Error (MSE), Cross-Entropy.

### **Learning Rate** ($\alpha$)
**What it is:** The step size.
*   **Big $\alpha$:** Fast learning, but might miss the target.
*   **Small $\alpha$:** Slow learning, but very precise.

---

## **A**

### **Activation Function**
**What it is:** A gatekeeper. It decides if a neuron should "fire" or stay silent.
*   **ReLU:** "If negative, become zero." (Filters noise).
*   **Sigmoid:** "Squash everything between 0 and 1." (Probabilities).

---

## **O**

### **Overfitting (High Variance)**
**What it is:** Memorizing the training data instead of learning the pattern.
*   **Symptom:** AI gets 100% on the Practice Exam, but 0% on the Real Exam.

---

## **T**

### **Token**
**What it is:** The atom of Language Models. It is a chunk of text (word or part of a word).
*   **LLM View:** ChatGPT doesn't see "Apple". It sees "App" + "le".
