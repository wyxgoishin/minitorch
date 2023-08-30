# MiniTorch
## Q&A
Q: No module named 'altair.vegalite.v4'

A: You have installed the wrong version of altair, try reinstall altair with specific version, e.g., 
`pip install altair==4.0`

Q: "ModuleNotFoundError: No module named `xxx`" when run `streamlit run app.py -- 0` in Windows

A: This is due to the lack of python path, try `python -m streamlit run app.py -- 0` instead

Q: I can not see specific error when running some test 
but got 'exceptiongroup.ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)' instead

A: This is a known [issue](https://github.com/HypothesisWorks/hypothesis/issues/3430) with hypothesis, 
try to use native python traceback with cmd as `pytest --tb=native xxx`