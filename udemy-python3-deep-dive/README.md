# Python Deep Dive

Instructor: [Fred Baptiste](https://saxobank.udemy.com/user/fredbaptiste/), professional developer and mathematician.

Many of these require Python 3.6 or above.

- [Part 1](https://www.udemy.com/course/python-3-deep-dive-part-1/?referralCode=E46B931C71EE01845062/): Mainly functional programming (Variables, Functions and Functional Programming, Closures, Decorators)
- [Part 2](https://www.udemy.com/course/python-3-deep-dive-part-2/?referralCode=3E7AFEF5174F04E5C8D4/): Sequences, Iterables, Iterators, Generators, Context Managers and Generator-based Coroutines
- [Part 3](https://www.udemy.com/course/python-3-deep-dive-part-3/?referralCode=C5B0D9AB965B9BF4C49F/): Hash Maps, Dictionaries, Sets, and Related Data Structures
- [Part 4](https://www.udemy.com/course/python-3-deep-dive-part-4/?referralCode=3BB758BE4C04FB983E6F/): OOP

## Part 1

This is Part 1 of a series of courses intended to dive into the inner mechanics and more complicated aspects of Python 3.

This is not a beginner course - if you've been coding Python for a week or a couple of months, you probably should keep writing Python for a bit more before tackling this series.

On the other hand, if you're now starting to ask yourself questions like:

- I wonder how this works?
- is there another way of doing this?
- what's a closure? is that the same as a lambda?
- I know how to use a decorator someone else wrote, but how does it work? Can I write my own?
- why isn't this boolean expression returning a boolean value?
- what does an import actually do, and why am I getting side effects?
- and similar types of question...

then this course is for you.

In this course series, I will give you a much more fundamental and deeper understanding of the Python language and the standard library.

Python is called a "batteries-included" language for good reason - there is a ton of functionality in base Python that remains to be explored and studied.

So this course is not about explaining my favorite 3rd party libraries - it's about Python, as a language, and the standard library.

In particular this course is based on the canonical CPython.
You will also need Jupyter Notebooks to view the downloadable fully-annotated Python notebooks.

It's about helping you explore Python and answer questions you are asking yourself as you develop more and more with the language.

In Python 3: Deep Dive (Part 1) we will take a much closer look at:

- Variables - in particular that they are just symbols pointing to objects in memory
- Namespaces and scope
- Python's numeric types
- Python boolean type - there's more to a simple or statement than you might think!
- Run-time vs compile-time and how that affects function defaults, decorators, importing modules, etc
- Functions in general (including lambdas)
- Functional programming techniques (such as map, reduce, filter, zip, etc)
- Closures
- Decorators
- Imports, modules and packages
- Tuples as data structures
- Named tuples

To get the most out of this course, you should be prepared to pause the coding videos, and attempt to write code before I do!
Sit back during the concept videos, but lean in for the code videos!

## Part 2

Part 2 of this Python 3: Deep Dive series is an in-depth look at:

- sequences
- iterables
- iterators
- generators
- comprehensions
- context managers
- generator based coroutines

I will show you exactly how iteration works in Python - from the sequence protocol, to the iterable and iterator protocols, and how we can write our own sequence and iterable data types.

We'll go into some detail to explain sequence slicing and how slicing relates to ranges.

We look at comprehensions in detail as well and I will show you how list comprehensions are actually closures and have their own scope, and the reason why subtle bugs sometimes creep in to list comprehensions that we might not expect.

We'll take a deep dive into the itertools module and look at all the functions available there and how useful (but overlooked!) they can be.

We also look at generator functions, their relation to iterators, and their comprehension counterparts (generator expressions).

Context managers, an often overlooked construct in Python, is covered in detail too. There we will learn how to create and leverage our own context managers and understand the relationship between context managers and generator functions.

Finally, we'll look at how we can use generators to create coroutines.

Each section is followed by a project designed to put into practice what you learn throughout the course.

This course series is focused on the Python language and the standard library.
There is an enormous amount of functionality and things to understand in just the standard CPython distribution, so I do not cover 3rd party libraries - this is a Python deep dive, not an exploration of the many highly useful 3rd party libraries that have grown around Python - those are often sufficiently large to warrant an entire course unto themselves!
Indeed, many of them already do!

## Part 3

This course is an in-depth look at Python dictionaries.

Dictionaries are ubiquitous in Python.
Classes are essentially dictionaries, modules are dictionaries, namespaces are dictionaries, sets are dictionaries and many more.

In this course we'll take an in-depth look at:

- associative arrays and how they can be implemented using hash maps
- hash functions and how we can leverage them for our own custom classes
- Python dictionaries and sets and the various operations we can perform with them
- specialized dictionary structures such  as OrderedDict and how it relates to the built-in Python3.6+ dict
- Python's implementation of multi-sets, the Counter class
- the ChainMap class
- how to create custom dictionaries by inheriting from the UserDict class
- how to serialize and deserialize dictionaries to JSON
- the use of schemas in custom JSON deserialization
- a brief introduction to some useful libraries such as JSONSchema, Marshmallow, PyYaml and Serpy

## Part 4

This Python3: Deep Dive Part 4 course takes a closer look at object oriented programming (OOP) in Python.

- what are classes and instances
- class data and function attributes
- properties
- instance, class and static methods
- polymorphism and the role special functions play in this
- single inheritance
- slots
- the descriptor protocol and its relationship to properties and functions
- enumerations
- exceptions
- metaprogramming (including metaclasses)
