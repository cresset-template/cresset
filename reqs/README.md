# Requirements

This directory contains `conda`, `apt`, and `pip` 
requirements files for the Dockerfiles.

Using requirements files should minimize 
the need to manually edit the Dockerfiles.

Note that the current project structure only allows the Dockerfile to find
requirements files in the `reqs` directory and project root directory because of the `.dockerignore` file.

To use files in other directories, 
please modify the `.dockerignore` file.

## Build Dependency Versions

Edit the package versions in `*-build.requirements.txt` if the latest versions cannot be used for older versions of PyTorch and other libraries.

`Setuptools` must be set to `<=59.5.0` for PyTorch `v1.10.x` and below.

`PyYAML` may cause issues for early versions of PyTorch.

More versioning issues will arise with the passing of time but the latest versions of libraries will use the latest versions of their dependencies.


## Requirements File Explanation

```
sed 's/#.*//g; s/\r//g' FILE | xargs -r COMMAND
```

Arbitrary commands can be executed from input text files with the syntax above.
For the Cresset project, this technique was used to create requirements files 
for `apt`, which does not natively support them. 
However, this technique will prove invaluable for many applications, 
which motivates this guide.

The `sed 's/#.*//g; s/\r//g' FILE` reads `FILE`, 
removes all comments, which start with a hash symbol, using `s/#.*//g`,
then converts all line endings to `\n` by removing `\r`.

The output of `sed` is given to `stdout`, which is the piped to `xargs`.
The `-r` flag stops execution if no inputs are given.
`xargs` takes all whitespace separated inputs from `stdin` and
runs `COMMAND` on these inputs. 
The newline `\n` character is used as a separator. 
Having multiple newlines in succession does not affect the result. 

The final result is that `COMMAND` is executed on all elements in 
each line of `FILE` while ignoring comments and blank lines.

Note that spaces in a single line may lead to bugs by splitting the line.
To use spaces, check the `xargs` documentation on how it handles whitespace.

# Adding Custom Code

The project blocks any files other than requirements files from being included in the Dockerfile context.
To add custom code not available from a download, edit the `.dockerignore` file to include the directory in the context.
Then `COPY` the directory in the Dockerfile during the build.
