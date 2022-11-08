{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN74YoaUwT2GE9zRw/RITgd",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mayurjadhav2002/practicals/blob/main/apriority.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 373
        },
        "id": "jHKAsfTM1XCy",
        "outputId": "5e61f9e6-a21f-41e0-8b50-5509505f9f5e"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-680aa7b79ed4>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mapyori\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mapyori\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mapriori\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mmlxtend\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfrequent_patterns\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mapriori\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0massociation_rules\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'apyori'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import apyori\n",
        "from apyori import apriori\n",
        "from mlxtend.frequent_patterns import apriori, association_rules"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.read_excel('/apriori.xlsx')\n",
        "frq_items = apriori(data, min_support = 0.6, use_colnames = True)\n",
        "rules = association_rules(frq_items, metric = \"confidence\", min_threshold = 0.80) \n",
        "print(rules)"
      ],
      "metadata": {
        "id": "1F-GLDOj1Z1i"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}