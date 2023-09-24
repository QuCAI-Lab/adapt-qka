<div align="center">
  <a href="#"><img valign="middle" height="45px" src="https://img.icons8.com/puzzle" width="45" hspace="0px" vspace="0px"> <h1> Style Guide </h1></a>
</div>
<br>

# [Python Naming Conventions](https://pep8.org/#prescriptive-naming-conventions) - Cheat Sheet

* Avoid single-letter variable names such as ‘l’ (lowercase letter el), ‘O’ (uppercase letter oh), or ‘I’ (uppercase letter eye) characters as these can be mistaken for 1 and 0 depending on the typeface.
* Use lowercase and snake_case (underscore_separated) conventions for function names and variable names as in `some_function` and `some_variable`, respectively.
* Use lowercase and snake_case conventions for method names as in `method` or `some_method`, respectively.
* Use lowercase and snake_case conventions for module names as in `module.py` or `some_module.py`, respectively.
* Use PascalCase (CapWords) convention for class names as in `ClassName` or `SomeClass`.
* Use only capital letters (uppercase) for constants as in `CONSTANT` or `SOME_CONSTANT`.


# Code Formatting

## Indentation

- Use 2 spaces per indentation level.

```python
def some_function(arg1: object) -> bool:
  '''
  Returns True if the value is None or empty. Returns False otherwise.  
  '''
  if not arg1: # 1st identation level.
    return True # 2nd identation level.
  return False
```

## Backslash

- Use a backslash (escape character) to break lines that are just too long:

```python
from some_package import example1, example2 \
  example3
```

## [F-strings](https://realpython.com/python-f-strings/#f-strings-a-new-and-improved-way-to-format-strings-in-python)

- Use Python f-strings to speed up readability over %-formatting and str.format().

Example with a Python expression:
```python
method_name = 'Some_Method'
f"{method_name.lower()} is the correct convention."
```

## Leading and trailing whitespace/characters

Remove spaces at the beginning (left) and at the end (right) of a string variable using Python's strip() method:

```python
string_variable= f" some text  "
string_variable = string_variable.strip()
```
```python
>>> 'some text'
```
Remove Leading and trailing characters of a string variable:

```python
string_variable= f",##,,,some text...rr..."
string_variable = string_variable.strip(",.#r")
```
```python
>>> 'some text'
```

## General rules for Docstrings

This project standard is built in close resemblance to the [pandas docstring guide](https://pandas.pydata.org/docs/development/contributing_docstring.html).

- The short summary of the function, method, or class must start with a capital letter, end with a dot, and fit in a single line.
- Use the backtick symbol (\`) to convey a Python variable, function, method, class, or module.
- Remove blank lines at the opening and closing quotes of the docstring. i.e., remove all blank lines after the signature `def func():`.
- The opening and closing quotes must be kept isolated from the description message.
- Use parentheses after the variable name to indicate its data type.

**Good example:**
```python
from typing import Union
def sum_ab(a: Union[float, str, list, tuple], b: Union[float, str, list, tuple]) -> Union[float, str, list, tuple]:
  """
  Adds two objects together.

  Args:
    a (Union[float, str, list, tuple]): The first object to be added.
    b (Union[float, str, list, tuple]): The second object to be added.

  Returns:
    Union[float, str, list, tuple]: The sum of objects `a` and `b`.

  Examples:
    >>> sum_ab(2, 2)
    4
    >>> sum_ab([1, 2], [3, 4])
    [1, 2, 3, 4]
  """
  variable_name = a + b
  return variable_name
```

# Argparse

- Use Python's built-in Argparse module to enable the user to set optional command line arguments/instructions (a.k.a environment variables) from the CLI to speed up code changes.

# Single and double underscores

- Use single underscore `_` to indicate a temporary variable that will not be used as an active index in a for loop such as in: 
```python 
for _ in range(n):
  # some_code
```
     
- Use single trailing underscore `foo_` to avoid conflict with a python reserved keyword such as in: `in_`, `complex_`, `def_`, or `class_`.

- Use single leading underscore `_foo` to indicate non-public methods and instance variables, i.e., accessible only in the local namespace (in the current script) that will not be available outside through wildcard import (`from module import *`), but only directly such as in: 
```python 
from module import _foo
```
or 
```python 
import module
module._foo
```    

- Use double leading and trailing underscore `__bar__` to: define special overwritable **D**ouble **Under**score (dunder) methods such as: `__init__`, `__str__`, `__repr__`, `__call__`, `__add__` and so on.

- Use double-leading underscore `__bar` to avoid name mangling, i.e., when an attribute of the base class is overwritten by an attribute of the subclass that is inheriting from the base class. Example usage: 
```python 
class Base:
  def __init__(self):
    self.__foo=value
object1=Base()
print(dir(object1))
```    
```python 
class Subclass(Base):
  def __init__(self):
    super(Subclass, self).__init__()
    self.__foo=value
object2=Subclass()
print(dir(object2))
```    
# Dunder methods

- Use Python's `__str__` and `__repr__` dunder methods to overwrite the print statement for class objects and to add user and developer information, respectively. Usage:

```python
class Complex_(object):
  def __init__(self, real:float=None, imag:float=None) -> None:
    self.real=real
    self.imag=imag

  def __complex__(self):
    '''
    Return the corresponding complex number.
    '''
    return self.real_new + self.imag_new*1j

  def __add__(self, other):
    '''Add two objects (complex numbers).'''
    self.name = "__add__"
    self.real_new = self.real + other.real
    self.imag_new = self.imag + other.imag
    return self

  def __sub__(self, other):
    '''Subtract two objects (complex numbers).'''
    self.name = "__sub__"
    self.real_new = self.real - other.real
    self.imag_new = self.imag - other.imag
    return self

  def __mul__(self, other):
    '''Multiply two objects (complex numbers).'''
    self.name = "__mul__"
    self.real_new = (self.real * other.real) - (self.imag * other.imag)
    self.imag_new = (self.real * other.imag) + (self.imag * other.real)
    return self

  def __truediv__(self, other):
    '''Divide two objects (complex numbers).'''
    self.name = "__truediv__"
    denominator = (other.real ** 2) + (other.imag ** 2)
    self.real_new = ((self.real * other.real) + (self.imag * other.imag))/denominator
    self.imag_new = ((self.imag * other.real) - (self.real * other.imag))/denominator
    return self

  def __str__(self):
    '''
    Return a human-readable string representation of the class for users.
    '''
    return str(self.__complex__())
  
  def __repr__(self):
    '''
    Return a human-readable string representation of the class for developers.
    '''
    return f"Result = {self.__complex__()}, {self.name} method was used."

z=Complex_(1,2)
print(f'{z+z}, {z-z}, {z*z}, {z/z}\n')
print(f'{repr(z+z)}\n{repr(z-z)}\n{repr(z*z)}\n{repr(z/z)}')
```
