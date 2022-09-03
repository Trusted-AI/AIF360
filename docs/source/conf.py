# -*- coding: utf-8 -*-
#
# aif360-bias-toolbox documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  8 14:05:59 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import aif360
import fairlearn


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax']

intersphinx_mapping = {'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'fairlearn': (f'https://fairlearn.github.io/v{fairlearn.__version__}/', None),
    'python': ('https://docs.python.org/{}.{}'.format(*sys.version_info), None)}

napoleon_include_init_with_doc = True
napoleon_use_ivar = True
napoleon_use_rtype = False

autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# generate autosummary even if no references
autosummary_generate = True

autodoc_default_flags = [
    'members',
    # 'inherited-members'
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'aif360'
copyright = f'2018 - {datetime.now().year}, The AI Fairness 360 (AIF360) Authors'
author = 'aif360 developers'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = aif360.__version__
# The short X.Y version.
version = '.'.join(release.split('.')[:2])

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['templates']

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'literal'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'bizstyle'
if os.environ.get('READTHEDOCS') != 'True':
    try:
        import sphinx_rtd_theme
    except ImportError:
        pass  # assume we have sphinx >= 1.3
    else:
        html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

def setup(app):
    app.add_css_file('style.css')

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
# html_sidebars = {
#     '**': [
#         'about.html',
#         'navigation.html',
#         'relations.html',  # needs 'show_related': True theme option to display
#         'searchbox.html',
#         'donate.html',
#     ]
# }


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'aif360doc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'aif360.tex', u'AI Fairness 360 Documentation',
     u'IBM Research', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'aif360', u'AI Fairness 360 Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'aif360', u'AI Fairness 360 Documentation',
     author, 'aif360', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for linkcode -------------------------------------------------
# taken from numpy/doc/source/conf.py:
import inspect
from os.path import relpath, dirname
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    fn = relpath(fn, start=dirname(aif360.__file__))

    return "https://github.com/Trusted-AI/AIF360/blob/master/aif360/%s%s" % (
           fn, linespec)
