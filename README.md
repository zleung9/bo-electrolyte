# Introduction
1. The intention of this repo is to serve as "the Automat repo" that hosts new API including lab, database, data QA/QC, ML and automation, etc.
2.  Development branch is now `dev`, individual contributions will be merged to this branch. Major version changes happen on `main` branch.
3.  The new structure has a  `docs`  folder that will automatically extract docstring from function/class definitions and create a documentation page. We can also add tutorial pages/ workflow pages to it.

# Repo structure
The repo structure folllows the suggestion by **Cookiecutter Data Science** from this [link](https://drivendata.github.io/cookiecutter-data-science/):

>When we think about data analysis, we often think just about the resulting reports, insights, or visualizations. While these end products are generally the main event, it's easy to focus on making the products look nice and ignore the quality of the code that generates them. Because these end products are created programmatically, code quality is still important! And we're not talking about bikeshedding the indentation aesthetics or pedantic formatting standards — ultimately, data science code quality is about correctness and reproducibility.
>
>It's no secret that good analyses are often the result of very scattershot and serendipitous explorations. Tentative experiments and rapidly testing approaches that might not work out are all part of the process for getting to the good stuff, and there is no magic bullet to turn data exploration into a simple, linear progression.
>
>That being said, once started it is not a process that lends itself to thinking carefully about the structure of your code or project layout, so it's best to start with a clean, logical structure and stick to it throughout. We think it's a pretty big win all around to use a fairly standardized setup like this one. 

# Installation
conda env:
`/data/miniconda3/envs/btch_dp`

# Run the script
To run:
> python -m src.TURBO_generation

# Documentation
To update the documentation page, go to `docs` and run: `make html`. A list of `html` file will be created in `built/html` folder. 
To view the documentation, open the `index.html` using a web browser.



**chemicals**
In  `chemicals/chemical.py`  There is a class for chemicals. You are welcome to define the properties for chemicals to store to MongoDB during your research.