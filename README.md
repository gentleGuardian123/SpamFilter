## Run the code

1. Create and activate the virtualenv:
  - linux: 
  ```
  $ python3 -m venv <myenvname> && source ./<myvenvname>/bin/activate
  ```

  - windows: 
  ```
  $ python3 -m venv <myenvname> 
  $ ./<myvenvname>/bin/activate.bat
  ```

2. Install python requirements:
  ```
  $ pip install -r requirements.txt
  ```

3. Install NLTK data:
  ```
  $ python3
  >>> import nltk
  >>> nltk.download('stopwords')
  >>> nltk.download('wordnet')
  >>> quit()
  ```

4. Execute the `main.py`:
  ```
  $ python3 main.py
  ```

5. The output file could be checked at `path/to/SpamFilter/Results/results.txt`.
