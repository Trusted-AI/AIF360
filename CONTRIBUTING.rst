######################
Contributing to AIF360
######################
Thank you for your interest in contributing to AIF360! Please read the document
below for an overview of the contribution process and different ways you may be
able to contribute. If you have a question not covered here, please check
`GitHub <https://github.com/Trusted-AI/AIF360/issues>`_ or ``#aif360-developers``
on `Slack <https://aif360.slack.com>`_ first.

Setup for development
=====================
Only maintainers with write access are able to create branches in the main
repository. All other contributors must create a fork of the repository from their
personal GitHub account and make a
`pull request back upstream <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.
See the GitHub docs for instructions on how to
`Fork a repo <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.
Running ``git remote -v`` should look like this::

    origin  https://github.com/<username>/AIF360.git (fetch)
    origin  https://github.com/<username>/AIF360.git (push)
    upstream        https://github.com/Trusted-AI/AIF360.git (fetch)
    upstream        https://github.com/Trusted-AI/AIF360.git (push)

Then, you may install the local version of AIF360 as editable so you can test
changes you make without reinstalling by navigating to the project root and
running::

    pip install -e '.[all]'

If you are using the "original" (not scikit-learn compatible) API, you may also
want to download the datasets for testing. Instructions can be found in the
``aif360/data/README.md`` file.

Issues
======
Issues should be created for bugs or feature suggestions. For general usage
questions, please try Slack. In most cases, a pull request should close one or
more issues. If no issue exists, feel free to create an issue and corresponding
PR at the same time. If you see an open issue you wish to contribute to, please
leave a comment asking to be assigned to it so others know it is being worked
on.

Bug reports should be accompanied by a trace showing the error message along
with a short example reproducing the issue. If you are able to diagnose the
problem and suggest a fix that is especially helpful. If you're willing to
implement the change and submit a PR that is best!

If you're interested in contributing but don't know where to start, try filtering
the issues by tag: ``"good first issue"``, ``"help wanted"``,
``"contribution welcome"``, or ``"easy"``.

Documentation
=============
Changes to documentation only (e.g., clarifying descriptions, improving the user
guide) should be discussed in an issue first. A PR which fixes obvious typos
does not require a corresponding issue.

All new functions or classes should include a docstring specifying inputs and
outputs and their respective types, a short (and optionally, long) description,
and an example snippet, if non-obvious. AIF360 uses
`Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

To generate the documentation using Sphinx, you will first need to install the
``[docs]`` extras if you have not already (e.g., via ``pip install -e '.[all]'``).
Then, run::

    make html

from the ``docs/`` directory and review the resulting html files in
``docs/build/html/``. In particular, make sure any new
pages are rendered properly, have no broken links, and match existing pages in
thoroughness and style.

User guide
----------
We would like to create a full user guide explaining the capabilities of AIF360
similar to `scikit-learn <https://scikit-learn.org/stable/user_guide.html>`_ If
you wish to contribute to this, please reach out
`here <https://github.com/Trusted-AI/AIF360/issues/426>`_.

Website
-------
Any comments or suggestions regarding the
`AIF360 homepage <https://aif360.res.ibm.com/>`_ may be posted to GitHub as
well. This content may eventually be merged with the rest of the documentation.

Example Notebooks
=================
All new algorithms or metrics should be accompanied by an example notebook
demonstrating a typical use case or reproducing experiments from the paper. The
notebook should contain text explaining major steps (assume an educated but not
expert audience). Results should be reliable and ideally reproducible via random
seed and demonstrate a clear advantage. This should be more involved than unit
tests and may require increased computational time. Datasets should be easy to
load from the notebook without manually downloading and extracting from a link, if
possible, and the license and any terms should be clearly presented. If the
dataset may be of general use to the community, consider contributing a script for
``aif360/sklearn/datasets`` as well. Finally, keep the code simple -- if using a
popular external library instead of writing custom functions is cleaner, go ahead
(just remember to add it to the notebook requirements in ``setup.py``) -- but
thorough.

Examples should be placed in the ``examples/`` or ``examples/sklearn/`` directory
depending on if they apply to the original or scikit-learn compatible API,
respectively. They should be named as ``demo_<class_name>.ipynb`` according to the
feature they demonstrate.

Tests
=====
Tests should be fast (< 15s) and deterministic but avoid overfitting to a random
seed. If the assertion result is not obvious, a comment should explain how it was
calculated. Remember: these tests exist to catch implementation mistakes, test
edge/corner cases, check expected error cases, validate inputs/outputs with other
parts of the toolkit, etc. -- example notebooks should demonstrate efficacy and
real-life use cases.

Tests are run with ``pytest``::

    pytest tests/

See `How to invoke pytest <https://docs.pytest.org/en/stable/how-to/usage.html>`_
for advanced usage.

Coverage for new code should exceed 80%. This can be checked using
``pytest-cov``::

    pytest tests/ --cov=aif360

Tests live in the ``tests/`` or ``tests/sklearn/`` directory (**not** with the
source files) depending on the version of interface they target. ``conftest.py``
containts fixtures that may be used by any test within that directory (see the
`pytest docs <https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files>`_
for details).

Datasets
========
AIF360 does not host datasets directly in the repository. If you wish to
contribute a data loading script which downloads and caches a dataset
automatically, please ensure that the license is permissive (not copyleft) and any
terms are clearly presented to the user before downloading.

New Features
============
New feature contributions will likely involve multiple of the sections above. For
example, a new algorithm will require an issue, documentation, an example
notebook, unit tests, and possibly a dataset. Please review this section and the
`PR checklist`_ so your code can be merged in a timely manner.

References
----------
New feature requests should be already studied in a scientific paper. Ideally,
these should be peer-reviewed (or under review) and open-access (or provide a
public link). A link to the paper should be added to the docstring under the
"References:" section and probably in the example notebook as well.

Coding conventions
------------------
Try to conform to the
`scikit-learn guidelines <https://scikit-learn.org/stable/developers/develop.html>`_
on developing estimators. Also, see the :ref:`Getting Started <sklearn-api>`
page for examples of when this style may be broken. If your case does not fit any
existing examples, please start a discussion on the issue or PR.

For deep learning algorithms, please use PyTorch (or alternatively, TensorFlow)
unless there is a good reason not to.

Also, don't forget to add an import for your class/function to the submodule's
``__init__.py`` so top-level functions/classes can be imported from the submodule
directly. Avoid ``import *``, whenever possible.

PR Checklist
============
Code
----
Remember to remove unnecessary imports, print statements, and commented code. If
any code is copy-pasted from somewhere else, make sure to attribute the source.
All added files should be human-readable (no binary files) except example
notebooks/images. Any necessary pre-trained models or data should be downloaded
from a (*trusted*) external source.

Naming, description
-------------------
Please be descriptive when creating a PR but also remember that the code should
speak for itself -- it should be readable with good commenting and documentation.
The description should explain the high-level changes, reference the inciting
issue, mention the license of any new libraries/datasets, and note any
compatibility issues that might arise. This is also a place to leave questions for
discussion with the reviewer.

Draft/WIP
---------
For larger contributions, it may be useful to create a draft PR containing
work-in-progress. In this case, please specify if you want feedback from the
maintainers since by default they will only review PRs which are marked ready
for review and have no merge issues.

Testing, examples, documentation
--------------------------------
Pull requests contributing new features (e.g., metrics, algorithms) must include
`unit tests <#tests>`_. If an existing test is failing, the fix does not require
any new tests but a bug not caught by any test should have a new test submitted
along with the fix.

New features should also be accompanied by an `example notebook <#example-notebooks>`_.

Also remember to add a line to the corresponding .rst file in ``docs/source/modules/``
so an autosummary will be generated and displayed in the
`documentation <#documentation>`_.

Link to issue, tag relevant maintainer
--------------------------------------
A PR should close at least one relevant issue. If no issue exists yet, just
submit the issue and PR at the same time. PRs and issues may be linked by using
`closing keywords <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword>`_
in the description or via the sidebar on the right.

Feel free to assign a `maintainer <#maintainers>`_ to review the changes if they
are the last significant contributor to the relevant code. For new code, you may
tag ``@hoffmansc`` or ``@nrkarthikeyan``. If there is no response for more than 7
days, please politely remind the reviewer.

DCO
---
This repository requires a
`Developer's Certificate of Origin 1.1 <https://elinux.org/Developer_Certificate_Of_Origin>`_
signoff on every commit. A DCO provides your assurance to the community that you
wrote the code you are contributing or have the right to pass on the code that
you are contributing. It is generally used in place of a Contributor License
Agreement (CLA). You can easily signoff a commit by using the `-s` or
`--signoff` flag::

    git commit -s -m 'This is my commit message'

If you are using the web interface, this should happen automatically. If you've
already made a commit, you can fix it by amending the commit and force-pushing
the change::

    git commit --amend --no-edit --signoff
    git push -f

This will only amend your most recent commit and will not affect the message. If
there are multiple commits that need fixing, you can try::

    git rebase --signoff HEAD~<n>
    git push -f

where `<n>` is the number of commits missing signoffs.

Branch protection
-----------------
Merging a pull request requires approval by at least one reviewer with write
access to the repository. There are also various automated checks which are run
including the DCO bot, `LGTM analysis <https://lgtm.com/projects/g/Trusted-AI/AIF360/>`_,
and Continuous Integration tests run through GitHub Actions. **Before** submitting
a PR or marking it as ready for review, please ensure tests and documentation
building run locally. If you don't know how to fix an error, you can mark the PR as
a draft and ask for help.

First-time contributors require approval to run workflows with GitHub actions. CI
should run unit tests for both Python and R for all supported versions as well as
print linter warnings. See
`ci.yml <https://github.com/Trusted-AI/AIF360/blob/main/.github/workflows/ci.yml>`_
for the latest build script.

Community
=========
Maintainers
-----------
The maintainers with write access are listed below in alphabetical order:

* Michael Hind (`michaelhind <https://github.com/michaelhind>`_)
* Samuel Hoffman (`hoffmansc <https://github.com/hoffmansc>`_)
* Anupama Murthi (`anupamamurthi <https://github.com/anupamamurthi>`_)
* Manish Nagireddy (`mnagired <https://github.com/mnagired>`_)
* Karthikeyan Natesan Ramamurthy (`nrkarthikeyan <https://github.com/nrkarthikeyan>`_)
* Stacey Ronaghan (`srnghn <https://github.com/srnghn>`_)
* Animesh Singh (`animeshsingh <https://github.com/animeshsingh>`_)
* Saishruthi Swaminathan (`SSaishruthi <https://github.com/SSaishruthi>`_)
* Gabriela de Queiroz (`gdequeiroz <https://github.com/gdequeiroz>`_)

Slack
-----
Discuss toolkit questions, fairness topics, and connect with the community on
`Slack <https://aif360.slack.com>`_!

`Join with this link <https://join.slack.com/t/aif360/shared_invite/zt-5hfvuafo-X0~g6tgJQ~7tIAT~S294TQ>`_

Monthly bee
-----------
We host a semi-regular meeting to bring the community together and provide a place
to discuss, plan, and learn. Meetings usually consist of a short talk on a
fairness topic, review of changes made in the last month, and open discussion on
the future roadmap for the next month and beyond. Connect on Slack in the
``monthly-bee`` channel to get notified about the next bee.
