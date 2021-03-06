![WebScripts Logo](https://mauricelambert.github.io/info/python/code/WebScripts/small_logo.png "WebScripts logo")

# WebScriptsClient

## Description

This package implements the "official" WebScriptsClient as a python module with a CLI.

Features implemented:

 - Get scripts, arguments and informations
 - Execute scripts on WebScripts Server
 - Download file from WebScripts Server
 - Upload file on WebScripts Server
 - Send requests or reports to WebScripts Administrator
 - Test the WebScripts Server

## Requirements

This package require:

 - python3
 - python3 Standard Library

## Installation

```bash
pip install WebScriptsClient
```

## Usages

### Command line

#### Module

```bash
python3 -m WebScriptsClient -h
```

#### Python executable

```bash
python3 WebScriptsClient.pyz --help
```

#### Command

```bash
# Tests
WebScriptsClient -v -u Admin -p Admin test http://127.0.0.1:8000/web/                                  # Test your WebScripts Server with verbose mode (log level DEBUG)

# Informations
WebScriptsClient info http://127.0.0.1:8000/web/                                                       # Script informations without authentication
WebScriptsClient -a AdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdmin info -a -c -d -s "test_config.py" "/auth/" http://127.0.0.1:8000/web/    # Get informations about test_config.py and /auth/ scripts without category, description and arguments (authenticated with API key)

# Execution
WebScriptsClient -A exec "test_config.py" http://127.0.0.1:8000/web/                                   # Run a script named test_config.py without arguments using the API key (prompt, interactive mode)
python -c "print('test')" | WebScriptsClient -u Admin -p Admin exec "test_config.py" -o test.txt -I http://127.0.0.1:8000/web/ # Run a script named test_config.py without arguments, using username and password for authentication, redirect output in file named test.txt and use STDIN as inputs
WebScriptsClient -u Admin -p Admin exec "test_config.py" -I test.txt http://127.0.0.1:8000/web/        # Run a script named test_config.py without arguments, using username and password for authentication and using test.txt as inputs
WebScriptsClient -u Admin -p Admin exec "test_config.py --test test3 --test4 -t" -i "test1" "test2" http://127.0.0.1:8000/web/ # Run a script named test_config.py with arguments, using username and password for authentication and using inputs

# Upload
WebScriptsClient -u Admin -p Admin upload -r 1000 -w 1000 -d 1000 -f test.txt test.txt http://127.0.0.1:8000/web/ # Upload a local file named test.txt on WebScripts Server with read, write and delete permissions equal to 1000 (group Admin in default WebScripts database) using the default Admin account. The file will be named test.txt on the WebScripts Server.
WebScriptsClient -u Admin -p Admin upload -6 -b -C -H -c dGVzdA== test.txt http://127.0.0.1:8000/web/             # Upload a file content on WebScripts Server using Base64 encoding, without compression, as binary and hidden file. The file will be named test.txt on the WebScripts Server.
python -c "print('test')" | WebScriptsClient -u Admin -p Admin upload test.txt http://127.0.0.1:8000/web/         # Upload a file content from STDIN on the WebScript Server. The file will be named test.txt on the WebScripts Server.

# Download
WebScriptsClient -u Admin -p Admin download -f "LICENSE.txt" "test.txt" http://127.0.0.1:8000/web/     # Download files (LICENSE.txt and test.txt) from WebScripts Server
WebScriptsClient -u Admin -p Admin download -s -f "LICENSE.txt" "test.txt" http://127.0.0.1:8000/web/  # Download files (LICENSE.txt and test.txt) from WebScripts Server and the save it locally with same names
WebScriptsClient -u Admin -p Admin download -o test.txt -f "LICENSE.txt" "test.txt" http://127.0.0.1:8000/web/ # Download files (LICENSE.txt and test.txt) from WebScripts Server and save it locally in test.txt (file concatenation)

# Request or report
WebScriptsClient -u Admin -P request -s title -n Maurice -r request -c 500 http://127.0.0.1:8000/web/  # Request or report to WebScripts Administrator using username and password (prompt, interactive mode), adding Subject, Name, Message and the HTTP error code
```

### Python script

```python
from WebScriptsClient import WebScriptsClient

client = WebScriptsClient("http://127.0.0.1:8000/web/", username="Admin", password="Admin", api_key="AdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdmin")
client.auth()

# Informations
scripts = client.get_scripts(refresh=False)

# Execution
arguments = client.args_command_to_webscripts(["--test", "test1", "test3"])
inputs = client.args_command_to_webscripts(["--test", "test1", "test3"], is_inputs=True)
for output, error, code in client.execute_script("test_config.py", arguments, inputs):
    print(output, end="")
print(f"Error code: {code}")
print(f"Error: {error}")

# Upload
client.upload("upload.txt", open("upload.txt"), no_compression=False, is_base64=False, hidden=False, binary=False, read_permissions=0, write_permissions=1000, delete_permissions=1000)

# Download
file = client.download("upload.txt", save=True)

# Request
client.request("Access", "I need access to test_config.py", "Maurice LAMBERT", error_code=500)
```

## Links

 - [Github Page](https://github.com/mauricelambert/WebScriptsClient/)
 - [Documentation](https://mauricelambert.github.io/info/python/code/WebScriptsClient.html)
 - [Pypi package](https://pypi.org/project/WebScriptsClient/)
 - [Python Executable](https://mauricelambert.github.io/info/python/code/WebScriptsClient.pyz)
 - [Windows Executable](https://github.com/mauricelambert/WebScriptsClient/releases/download/v0.0.3/WebScriptsClient.exe)

## Help

```text
python WebScriptsClient.py --help
usage: WebScriptsClient.py [-h] [-v] [-i] [-u USERNAME] [-p PASSWORD | -P] [-a API_KEY | -A] {info,exec,upload,download,request,test} ... url

This package is the "official" WebScripts client. This package implements client for default WebScripts features.

positional arguments:
  {info,exec,upload,download,request,test}
                        Use default features of the WebScripts server.
    info                Get the available scripts, arguments, and information about them.
    exec                Execute a script on WebScripts Server.
    upload              Upload a file on WebScripts Server.
    download            Download files from WebScripts Server.
    request             Request or report to WebScripts administrator.
    test                Test WebScripts server and client.
  url                   URL of the WebScripts server, example: http://127.0.0.1:8000

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode (print logs).
  -i, --insecure        Do not check SSL certificate.
  -u USERNAME, --username USERNAME
                        WebScripts username to use for this connection.
  -p PASSWORD, --password PASSWORD
                        WebScripts password to use for this connection.
  -P, --password-prompt
                        Interactive password prompt.
  -a API_KEY, --api-key API_KEY
                        WebScripts password to use for this connection.
  -A, --api-key-prompt  Interactive API key prompt.
```

```text
python WebScriptsClient.py info --help
usage: WebScriptsClient.py info [-h] [-a] [-d] [-c] [-s SCRIPTS [SCRIPTS ...]]

optional arguments:
  -h, --help            show this help message and exit
  -a, --no-arguments    Do not print arguments.
  -d, --no-descriptions
                        Do not print description.
  -c, --no-categories   Do not print categories.
  -s SCRIPTS [SCRIPTS ...], --scripts SCRIPTS [SCRIPTS ...]
                        Information about specific scripts.
```

```text
python WebScriptsClient.py exec --help
usage: WebScriptsClient.py exec [-h] [-o [OUTPUT_FILENAME]] [-i INPUTS [INPUTS ...] | -I [INPUT_FILENAME]] command

positional arguments:
  command               Command to execute script on WebScripts Server.

optional arguments:
  -h, --help            show this help message and exit
  -o [OUTPUT_FILENAME], --output-filename [OUTPUT_FILENAME], --output [OUTPUT_FILENAME]
                        Output file to save the result.
  -i INPUTS [INPUTS ...], --inputs INPUTS [INPUTS ...]
                        Inputs value for script inputs.
  -I [INPUT_FILENAME], --input-filename [INPUT_FILENAME], --input [INPUT_FILENAME]
                        Input file for script inputs.
```

```text
python WebScriptsClient.py upload --help
usage: WebScriptsClient.py upload [-h] [-6] [-C] [-H] [-b] [-r READ_PERMISSION] [-w WRITE_PERMISSION] [-d DELETE_PERMISSION] [-f FILE | -c CONTENT] filename

positional arguments:
  filename              The file name of the uploaded file.

optional arguments:
  -h, --help            show this help message and exit
  -6, --base64, --64    Upload a base64 encoded file on the server.
  -C, --no-compression  Do not compress the file on the server.
  -H, --hidden          Hide the file on the server.
  -b, --binary, --bin   Upload a binary file on the server.
  -r READ_PERMISSION, --read-permission READ_PERMISSION, --read READ_PERMISSION
                        Read permission on the server.
  -w WRITE_PERMISSION, --write-permission WRITE_PERMISSION, --write WRITE_PERMISSION
                        Write permission on the server.
  -d DELETE_PERMISSION, --delete-permission DELETE_PERMISSION, --delete DELETE_PERMISSION
                        Delete permission on the server.
  -f FILE, --file FILE  The filename of the file to upload on the server.
  -c CONTENT, --content CONTENT
                        Content of the file.
```

```text
python WebScriptsClient.py download --help
usage: WebScriptsClient.py download [-h] -f FILENAMES [FILENAMES ...] [-o [OUTPUT_FILENAME]] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAMES [FILENAMES ...], --filenames FILENAMES [FILENAMES ...]
                        Filenames to download.
  -o [OUTPUT_FILENAME], --output-filename [OUTPUT_FILENAME], --output [OUTPUT_FILENAME]
                        Filename to write the downloaded content.
  -s, --save            Save the download in the same local filename.
```

```text
python WebScriptsClient.py request --help
usage: WebScriptsClient.py request [-h] [-s SUBJECT] [-n NAME] [-r REQUEST] [-c ERROR_CODE]

optional arguments:
  -h, --help            show this help message and exit
  -s SUBJECT, --subject SUBJECT
                        The request/report subject.
  -n NAME, --name NAME  Your name (Firstname LASTNAME).
  -r REQUEST, --request REQUEST
                        The request/report.
  -c ERROR_CODE, --error-code ERROR_CODE
                        The HTTP error code {403, 404, 406, 500 ...}.
```

```text
python WebScriptsClient.py test --help
usage: WebScriptsClient.py test [-h]

optional arguments:
  -h, --help  show this help message and exit
```

## Licence

Licensed under the [GPL, version 3](https://www.gnu.org/licenses/).
