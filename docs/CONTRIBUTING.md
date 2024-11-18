# Contributing to Biofit

Biofit is an open source project, so all contributions and suggestions are welcome.

You can contribute in many different ways: giving ideas, answering questions, reporting
bugs, proposing enhancements, improving the documentation, fixing bugs, and more.

Many thanks in advance to every contributor.

## How to Work on an Open Issue?

You have the list of open Issues at:
[Biofit Issues](https://github.com/psmyth94/biofit/issues)

## How to Create a Pull Request?

If you want to contribute to the codebase, follow these steps:

1. **Clone the Repository:**

   Clone the `dev` branch of the repository to your local disk:

   ```bash
   git clone git@github.com:psmyth94/biofit
   cd biofit
   ```

2. **Create a New Branch:**

   Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   Do not work on the `main` branch directly.

3. **Set Up a Development Environment:**

   Set up a development environment by running the following command:

   ```bash
   mamba env create -n biofit-local python=3.10
   mamba activate biofit-local
   pip install -e ".[test]"
   ```

   (If Biofit was already installed in the virtual environment, remove it with
   `pip uninstall biofit` before reinstalling it in editable mode with the `-e` flag.)

4. **Develop the Features on Your Branch:**

   Make your changes to the code.

5. **Format Your Code:**

   Format your code. Run `ruff` so that your newly added files look nice with the
   following command:

   ```bash
   ruff check . --fix
   ```

6. **(Optional) Use Pre-commit Hooks:**

   You can also use pre-commit to format your code automatically each time you run
   `git commit`, instead of running `ruff` manually. To do this, install pre-commit via
   `pip install pre-commit` and then run `pre-commit install` in the project's root
   directory to set up the hooks. Note that if any files were formatted by pre-commit
   hooks during committing, you have to run `git commit` again.

7. **Commit Your Changes:**

   Once you're happy with your contribution, add your changed files and make a commit
   to record your changes locally:

   ```bash
   git add -u
   git commit -m "Your commit message"
   ```

8. **Sync with the Original Repository:**

   It is a good idea to sync your copy of the code with the original repository
   regularly. This way you can quickly account for changes:

   ```bash
   git fetch upstream
   git rebase upstream main
   ```

9. **Push the Changes:**

   Once you are satisfied, push the changes to the remote repository using:

   ```bash
   git push origin a-descriptive-name-for-my-changes
   ```

10. **Create a Pull Request:**

    Go to the webpage of the repository on GitHub. Click on "Pull request" to send
    your changes to the project maintainers for review.

## Code of Conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you
are expected to abide by this code.
