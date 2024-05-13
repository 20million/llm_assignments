{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b2ba9157",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-05-13 20:44:44.394757: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
      "2024-05-13 20:44:54.333809: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "module 'tensorflow' has no attribute 'Session'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 8\u001b[0m\n\u001b[1;32m      4\u001b[0m b \u001b[38;5;241m=\u001b[39m tf\u001b[38;5;241m.\u001b[39mconstant(\u001b[38;5;241m3\u001b[39m)\n\u001b[1;32m      6\u001b[0m c \u001b[38;5;241m=\u001b[39m tf\u001b[38;5;241m.\u001b[39madd(a, b)\n\u001b[0;32m----> 8\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[43mtf\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mSession\u001b[49m() \u001b[38;5;28;01mas\u001b[39;00m sess:\n\u001b[1;32m      9\u001b[0m     result \u001b[38;5;241m=\u001b[39m sess\u001b[38;5;241m.\u001b[39mrun(c)\n\u001b[1;32m     10\u001b[0m     \u001b[38;5;28mprint\u001b[39m(result)\n",
      "\u001b[0;31mAttributeError\u001b[0m: module 'tensorflow' has no attribute 'Session'"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "a = tf.constant(2)\n",
    "b = tf.constant(3)\n",
    "\n",
    "c = tf.add(a, b)\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    result = sess.run(c)\n",
    "    print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7397a562",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "a = tf.constant(2)\n",
    "b = tf.constant(3)\n",
    "\n",
    "c = tf.add(a, b)\n",
    "\n",
    "result = c.numpy()\n",
    "print(result)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40e04802",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.16+"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
