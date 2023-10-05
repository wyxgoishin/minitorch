# MiniTorch
## Introduction
This is my implementation of [minitorch](https://minitorch.github.io/) 
([Cornell CS 5781](https://classes.cornell.edu/browse/roster/FA22/class/CS/5781)). All tasks implementation shall
be found in corresponding branches. Although all task branches can pass task no bigger than its tags, however, 
due to my incautious and the weakness of tests, some previous implementations
exists some subtle bugs that can affect latter tasks. In addition, some previous implementations are not so good enough.
So in case you want to borrow some hints from my code, you shall see the main branch which is the latest. 
And for several tasks, you shall see the cuda or fast version but not the simple version.
## Some Problems you may encounter
Q: No module named 'altair.vegalite.v4'

A: You have installed the wrong version of altair, try reinstall altair with specific version, e.g., 
`pip install altair==4.0`

Q: "ModuleNotFoundError: No module named `xxx`" when run `streamlit run app.py -- 0` in Windows

A: This is due to the lack of python path, try `python -m streamlit run app.py -- 0` instead

Q: I can not see specific error when running some test 
but got `exceptiongroup.ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)` instead

A: This is a known [issue](https://github.com/HypothesisWorks/hypothesis/issues/3430) with hypothesis, 
try to use native python traceback with cmd as `pytest --tb=native xxx`

Q: Task3.1 `test_two_grad_broadcast` failed with reason like `Examples routinely exceeded the max allowable size`

A: According to [numba_doc](https://hypothesis.readthedocs.io/en/hypothesis-python-4.57.1/healthchecks.html), this is due
to the problem of generation, so you can solve it with warning suppression like
```
@settings(suppress_health_check=[HealthCheck.data_too_large])
```

Q: IR version 1.6 incompatible with current version 2.0

A: This is due to the incompatibility of your numba version and cuda version. In almost all case your cuda version
may be too higher, then you can try to install `numba/label/dev/::numba` instead.

Q: Some dataset related issues in task4.5

A: Path of MNIST dataset shall be absolute path for windows. Dataset of glue/sst2 shall be downloaded from hugging face instead, as the key not exists in google oss now.