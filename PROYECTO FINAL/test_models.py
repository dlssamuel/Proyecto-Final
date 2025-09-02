{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fe652be0-fca5-4d95-b83e-8324540412e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Métricas de prueba con datos inventados:\n",
      "accuracy: 1.0\n",
      "precision: 1.0\n",
      "recall: 1.0\n",
      "f1_score: 1.0\n",
      "confusion_matrix: [[2 0]\n",
      " [0 1]]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from models import AlcoholLogisticModel\n",
    "\n",
    "# ===============================\n",
    "# 1. Crear dataset inventado\n",
    "# ===============================\n",
    "data = {\n",
    "    \"tipo_accidente\": [\"Colisión frontal\", \"Atropello\", \"Colisión frontal\", \"Choque múltiple\", \"Atropello\",\n",
    "                       \"Choque múltiple\", \"Colisión frontal\", \"Atropello\", \"Choque múltiple\", \"Colisión frontal\"],\n",
    "    \"tipo_vehiculo\": [\"Turismo\", \"Moto\", \"Bicicleta\", \"Turismo\", \"Moto\",\n",
    "                      \"Bicicleta\", \"Turismo\", \"Moto\", \"Turismo\", \"Bicicleta\"],\n",
    "    \"tipo_persona\": [\"Conductor\", \"Peatón\", \"Conductor\", \"Pasajero\", \"Conductor\",\n",
    "                     \"Conductor\", \"Peatón\", \"Pasajero\", \"Conductor\", \"Conductor\"],\n",
    "    \"sexo\": [\"Hombre\", \"Mujer\", \"Hombre\", \"Mujer\", \"Hombre\",\n",
    "             \"Hombre\", \"Mujer\", \"Mujer\", \"Hombre\", \"Mujer\"],\n",
    "    \"estado_meteorológico\": [\"Despejado\", \"Lluvia\", \"Nublado\", \"Despejado\", \"Lluvia\",\n",
    "                             \"Nublado\", \"Despejado\", \"Lluvia\", \"Nublado\", \"Despejado\"],\n",
    "    \"distrito\": [\"Centro\", \"Salamanca\", \"Chamartín\", \"Centro\", \"Salamanca\",\n",
    "                 \"Chamartín\", \"Centro\", \"Salamanca\", \"Chamartín\", \"Centro\"],\n",
    "    \"hora_num\": [23, 14, 3, 18, 22, 7, 15, 12, 2, 20],\n",
    "    \"rango_edad\": [\"18-30\", \"31-50\", \"51-65\", \"18-30\", \"31-50\",\n",
    "                   \"51-65\", \"18-30\", \"31-50\", \"51-65\", \"18-30\"],\n",
    "    \"positiva_alcohol\": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]\n",
    "}\n",
    "\n",
    "df_test = pd.DataFrame(data)\n",
    "\n",
    "# ===============================\n",
    "# 2. Separar X (predictoras) e y (objetivo)\n",
    "# ===============================\n",
    "X = df_test.drop(\"positiva_alcohol\", axis=1)\n",
    "y = df_test[\"positiva_alcohol\"]\n",
    "\n",
    "# One-hot encoding de categóricas\n",
    "X = pd.get_dummies(X, drop_first=True)\n",
    "\n",
    "# Escalar la variable numérica\n",
    "scaler = StandardScaler()\n",
    "if \"hora_num\" in X.columns:\n",
    "    X[[\"hora_num\"]] = scaler.fit_transform(X[[\"hora_num\"]])\n",
    "\n",
    "# ===============================\n",
    "# 3. Split train-test\n",
    "# ===============================\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.3, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# ===============================\n",
    "# 4. Usar nuestra clase\n",
    "# ===============================\n",
    "model = AlcoholLogisticModel(random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluación\n",
    "metrics = model.evaluate(X_test, y_test)\n",
    "print(\"Métricas de prueba con datos inventados:\")\n",
    "for k, v in metrics.items():\n",
    "    print(f\"{k}: {v}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
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
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
