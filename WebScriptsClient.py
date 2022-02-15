#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################
#    Official WebScripts client. Implements client for default WebScripts features.
#    Copyright (C) 2021  Maurice Lambert

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###################

"""
This package implements the "official" WebScripts client.
This package implements client for default WebScripts features.

Basic:

~# python -m WebScriptsClient -u Admin -p Admin exec "show_license.py" http://127.0.0.1:8000/web/auth/

ExitCode: 1
Errors: USAGE: show_license.py [part required string]

~# python -m WebScriptsClient -u Admin -p Admin exec "show_license.py copyright" http://127.0.0.1:8000/web/auth/

WebScripts  Copyright (C) 2021, 2022  Maurice Lambert
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.


ExitCode: 0
Errors:

~# python -m WebScriptsClient -u Admin -p Admin exec "show_license.py copyright error" http://127.0.0.1:8000/web/auth/

WebScripts  Copyright (C) 2021, 2022  Maurice Lambert
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.


ExitCode: 2
Errors: ERROR: unexpected arguments ['error']

Full examples:

~# python WebScriptsClient.py -v -u Admin -p Admin test http://127.0.0.1:8000/web/

~# python WebScriptsClient.py -v -u Admin -p Admin download -f "LICENSE.txt" "LICENSE.txt" http://127.0.0.1:8000/web/
~# python WebScriptsClient.py -v -u Admin -p Admin download -s -f "LICENSE.txt" "LICENSE.txt" http://127.0.0.1:8000/web/
~# python WebScriptsClient.py -v -u Admin -p Admin download -o test.txt -f "LICENSE.txt" "LICENSE.txt" http://127.0.0.1:8000/web/

-# python -c "print('test')" > test.txt
~# python WebScriptsClient.py -v -u Admin -p Admin upload -r 1000 -w 1000 -d 1000 -f test.txt test.txt http://127.0.0.1:8000/web/
~# python WebScriptsClient.py -v -u Admin -p Admin upload -6 -b -C -H -c dGVzdA== test.txt http://127.0.0.1:8000/web/
~# python -c "print('test')" | python WebScriptsClient.py -v -u Admin -p Admin upload test.txt http://127.0.0.1:8000/web/

~# python WebScriptsClient.py -v -u Admin -P request -s title -n Maurice -r request -c 500 http://127.0.0.1:8000/web/

~# python WebScriptsClient.py -v -A exec "test_config.py" http://127.0.0.1:8000/web/
~# python -c "print('test')" | python WebScriptsClient.py -v -u Admin -p Admin exec "test_config.py" -o test.txt -I http://127.0.0.1:8000/web/
~# python -c "print('test')" > test.txt
~# python WebScriptsClient.py -v -u Admin -p Admin exec "test_config.py" -I test.txt http://127.0.0.1:8000/web/
~# python WebScriptsClient.py -v -u Admin -p Admin exec "test_config.py --test test3 --test4 -t" -i "test1" "test2" http://127.0.0.1:8000/web/
~# python -m WebScriptsClient -u Admin -p Admin exec password_generator.py http://127.0.0.1:8000/web/auth/
~# python -m WebScriptsClient -u Admin -p Admin exec "show_license.py license" http://127.0.0.1:8000/web/auth/
~# python -m WebScriptsClient -u Admin -p Admin exec "show_license.py copyright codeheader" http://127.0.0.1:8000/web/auth/

~# python WebScriptsClient.py -v info http://127.0.0.1:8000/web/
~# python WebScriptsClient.py -a AdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdminAdmin info -a -c -d -s "test_config.py" "/auth/" http://127.0.0.1:8000/web/

>>> client = WebScriptsClient("http://127.0.0.1:8000/web/", username="user", password="pass", api_key="api key")
>>> client.auth()
>>> scripts = client.get_scripts(refresh=False)
>>> client.upload("upload.txt", open("upload.txt"), no_compression=False, is_base64=False, hidden=False, binary=False, read_permissions=0, write_permissions=1000, delete_permissions=1000)
>>> file = client.download("upload.txt", save=True)
>>> client.request("Access", "I need access to test_config.py", "Maurice LAMBERT", error_code=500)
>>> arguments = client.args_command_to_webscripts(["--test", "test1", "test3"])
>>> inputs = client.args_command_to_webscripts(["--test", "test1", "test3"], is_inputs=True)
>>> for output, error, code in client.execute_script("test_config.py", arguments, inputs): print(output, end="")
>>> print(f"Error code: {code}")
>>> print(f"Error: {error}")
"""

__version__ = "0.0.3"
__author__ = "Maurice Lambert"
__author_email__ = "mauricelambert434@gmail.com"
__maintainer__ = "Maurice Lambert"
__maintainer_email__ = "mauricelambert434@gmail.com"
__description__ = """
This package implements the "official" WebScripts client.
This package implements client for default WebScripts features.
"""
license = "GPL-3.0 License"
__url__ = "https://github.com/mauricelambert/WebScriptsClient"

copyright = """
WebScriptsClient  Copyright (C) 2022  Maurice Lambert
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.
"""
__license__ = license
__copyright__ = copyright

__all__ = [
    "WebScriptsClient",
    "WebScriptsError",
    "WebScriptsPermissionsError",
    "WebScriptsScriptNotFoundError",
    "WebScriptsAuthenticationError",
]

from urllib.request import (
    build_opener,
    Request,
    OpenerDirector,
    HTTPDefaultErrorHandler,
    HTTPRedirectHandler,
)
from platform import python_version, uname, python_implementation
from logging import StreamHandler, Formatter, Logger, getLogger
from argparse import ArgumentParser, Namespace, FileType
from typing import TypeVar, Tuple, List, Dict, Union
from http.client import HTTPResponse, HTTPMessage
from collections.abc import Callable, Iterator
from gzip import open as gzipopen, GzipFile
from ssl import _create_unverified_context
from sys import exit, stdout, argv, stdin
from urllib.response import addinfourl
from io import TextIOWrapper, BytesIO
from functools import wraps, partial
from tempfile import TemporaryFile
from urllib.parse import urlparse
from urllib.error import URLError
from operator import itemgetter
from shutil import copyfileobj
from json import load, dumps
from base64 import b64encode
from getpass import getpass
from shlex import shlex

Json = TypeVar("Json", dict, list, str, int, float, bool, None)


def get_custom_logger() -> Logger:

    """
    This function create a custom logger.
    """

    logger = getLogger(__name__)

    formatter = Formatter(
        fmt=(
            "%(asctime)s%(levelname)-9s(%(levelno)s) "
            "{%(name)s - %(filename)s:%(lineno)d} %(message)s"
        ),
        datefmt="[%Y-%m-%d %H:%M:%S] ",
    )
    stream = StreamHandler(stream=stdout)
    stream.setFormatter(formatter)

    logger.addHandler(stream)

    return logger


class __JsonPython:

    """
    This class create object and subobject from Json Structure.
    """

    def __init__(self, data: Json):
        logger_debug(f"Creating a {self.__class__.__name__}...")
        if isinstance(data, dict):

            logger_info("A dict is detected.")

            for key, value in data.items():
                if isinstance(value, dict):
                    logger_info("Launch a recursive instance.")
                    setattr(self, key.replace("-", "_"), NoName(value))
                else:
                    setattr(self, key.replace("-", "_"), value)

            logger_debug("Add __dict__ methods...")
            dict_ = self.__dict__
            for attribute in dir(data):
                if isinstance(getattr(data, attribute), Callable) and (
                    not attribute.startswith("__")
                    and not attribute.endswith("__")
                ):
                    setattr(self, attribute, getattr(dict_, attribute))
            logger_info("__dict__ methods is added.")


class Script(__JsonPython):
    pass


class Argument(__JsonPython):
    pass


class NoName(__JsonPython):
    pass


class WebScriptsError(Exception):
    pass


class WebScriptsPermissionsError(WebScriptsError):
    pass


class WebScriptsScriptNotFoundError(WebScriptsError):
    pass


class WebScriptsAuthenticationError(WebScriptsError):
    pass


class GETTER:

    """
    This class implements the optimized itemgetter.
    """

    output = itemgetter("stdout", "stderr", "code")


class WebScriptsOpener(HTTPDefaultErrorHandler, HTTPRedirectHandler):

    """
    This class implements a WebScripts opener to open URLs with urllib.
    """

    def http_error_403(
        self,
        request: Request,
        response: HTTPResponse,
        code: int,
        message: str,
        headers: HTTPMessage,
    ) -> None:

        """
        This function implements action on HTTP error 403.
        """

        log = f"HTTP error {code}: {message} -> you do not have permissions."
        url = request.get_full_url()
        command = (
            f"{argv[0]} [(-a <KEY>|-A|-u <USER> -p <PASSWORD>|-u -P)]"
            " request -n [name] -s Access -r"
            f' "I need access to this script." {url}'
        )
        logger_error(log)
        print(
            f"{log}\nYou can request access to the administrator "
            f"with:\n\t{command}"
        )

        raise WebScriptsPermissionsError(
            "You do not have access to this script."
        )

    def http_error_404(
        self,
        request: Request,
        response: HTTPResponse,
        code: int,
        message: str,
        headers: HTTPMessage,
    ) -> None:

        """
        This function implements action on HTTP error 404.
        """

        log = f"HTTP error {code}: {message} -> this script does not exists."
        logger_error(log)
        print(log)

        raise WebScriptsScriptNotFoundError("This script does not exists.")

    def http_error_302(
        self,
        request: Request,
        response: HTTPResponse,
        code: int,
        message: str,
        headers: HTTPMessage,
    ) -> None:

        """
        This function implements action on HTTP error 302
        used to redirect on the auth page.
        """

        log = f"HTTP error {code}: {message} -> you do not have permissions."
        url = request.get_full_url()
        command = (
            f"{argv[0]} [(-a <KEY>|-A|-u <USER> -p <PASSWORD>|-u -P)]"
            " request -n [name] -s Access -r"
            f' "I need access to this script." {url}'
        )
        logger_error(log)
        print(
            f"{log}\nYou can request access to the administrator "
            f"with:\n\t{command}"
        )

        raise WebScriptsPermissionsError(
            "You do not have access to this script."
        )


class WebScriptsAuthOpener(HTTPRedirectHandler):
    def http_error_302(
        self,
        request: Request,
        response: HTTPResponse,
        code: int,
        message: str,
        headers: HTTPMessage,
    ) -> addinfourl:

        """
        This function implements action on HTTP error 302
        used to redirect user on the Web page after authentication.
        """

        logger_debug("Capture redirect response (/auth/ -> /web/)...")
        return addinfourl(response, headers, request.get_full_url(), code)


opener: OpenerDirector = build_opener(WebScriptsAuthOpener)
authopener: Callable = opener.open
opener: OpenerDirector = build_opener(WebScriptsOpener)
urlopen: Callable = opener.open
logger: Logger = get_custom_logger()
logger_debug: Callable = logger.debug
logger_info: Callable = logger.info
logger_warning: Callable = logger.warning
logger_error: Callable = logger.error
logger_critical: Callable = logger.critical


class WebScriptsClient:

    """
    This class implements the "official" WebScripts client.
    """

    def __init__(
        self,
        url: str,
        username: str = None,
        password: str = None,
        api_key: str = None,
    ):

        logger_debug("Creating a WebScriptsClient...")
        logger_debug("Get URL...")
        url = urlparse(url)
        url = self.url = f"{url.scheme}://{url.netloc}"
        self.username = username
        self.password = password
        self.api_key = api_key
        logger_info("URL and credentials are save as attribute.")

        self.api_data = None
        self.scripts: List[Script] = None

        system = uname()
        headers = self.headers = {
            "User-Agent": (
                f"WebScriptsClient/{__version__} (Python"
                f"[{python_implementation()}]/{python_version()};"
                f" {system.system}/{system.release}) {system.node}"
            ),
            "Origin": url,
        }

        logger_debug("Build credentials...")
        if username is not None and password is not None:
            logger_debug("Build BasicAuth...")
            credentials = self.credentials = b64encode(
                f"{username}:{password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        if api_key is not None:
            logger_debug("Add Api-Key...")
            headers["Api-Key"] = api_key

    def auth(self, *args, **kwargs) -> None:

        """
        This function requests the /auth/ script to get a API Token.

        args and kwargs are sent to urllib.request.urlopen function.
        """

        logger_debug("Build request for /auth/ script...")
        username = self.username
        password = self.password
        api_key = self.api_key
        headers = self.headers
        arguments = None

        if username and password:
            logger_debug(
                "Build arguments with username and password and"
                " remove Authorization header..."
            )
            arguments = {"--username": username, "--password": password}
            del headers["Authorization"]

        if api_key:
            logger_debug(
                "Build arguments with API key and remove Api-Key header..."
            )
            arguments = {"--api-key": api_key}
            del headers["Api-Key"]

        if arguments is None:
            logger_warning(
                "No credentials found, do not perform authentication."
            )
            return None

        logger_debug("Send request for /auth/ script...")
        response = self.execute(
            "/auth/", arguments, {}, *args, urlopen=authopener, **kwargs
        )
        logger_debug("Response received.")
        cookie = response.headers["Set-Cookie"]
        logger_debug("Cookie found.")

        if cookie.startswith("SessionID=0:"):
            logger_error("Authentication error, your are not authenticated.")
            raise WebScriptsAuthenticationError(
                "Authentication error, your are not authenticated."
            )
        elif not cookie.startswith("SessionID="):
            raise WebScriptsAuthenticationError(
                "Received cookie is not valid."
            )

        headers["Api-Token"] = cookie
        logger_info("You are authenticated.")

    def get_scripts(
        self, *args, refresh: bool = False, **kwargs
    ) -> List[Script]:

        """
        This function requests WebScripts API to build Scripts and Arguments.
        """

        logger_debug(
            "Request /api/ (informations about scripts and arguments)."
        )
        api_data = self.api_data

        if refresh or not api_data:
            logger_debug("Send request.")
            response = urlopen(
                Request(f"{self.url}/api/", headers=self.headers),
                *args,
                **kwargs,
            )
            api_data = self.api_data = load(response)

        scripts = self.scripts = []

        for script_dict in api_data.values():
            arguments = script_dict.pop("args")
            script = Script(script_dict)
            script.arguments = [
                Argument(argument_dict) for argument_dict in arguments
            ]
            scripts.append(script)

        logger_info("Scripts object are built.")
        return scripts

    def execute_script(
        self,
        script_name: Union[Script, str],
        arguments: Dict[str, Json],
        inputs: Dict[str, Json],
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, str, int]]:

        """
        This function requests a WebScripts script and returns output.

        args and kwargs are sent to urllib.request.urlopen function.
        """

        if isinstance(script_name, Script):
            logger_debug("Get name from Script...")
            script_name = script_name.name

        logger_debug("Build the path of the script...")
        path = f"/api/scripts/{script_name}"

        response = self.execute(path, arguments, inputs, *args, **kwargs)
        data = load(response)
        yield GETTER.output(data)
        logger_info("First output is loaded and returned.")

        key = data.get("key")
        if key:
            yield from self.get_real_time_output(key, data, *args, **kwargs)

    def get_real_time_output(
        self, key: str, data: Dict[str, Json], *args, **kwargs
    ) -> Iterator[Tuple[str, str, int]]:

        """
        This function returns outputs of script with the "real time output"
        WebScripts feature.

        args and kwargs are sent to urllib.request.urlopen function.
        """

        request = Request(
            f"{self.url}/api/script/get/{key}", headers=self.headers
        )
        logger_info("Second request is built.")

        while "key" in data:
            logger_debug("Send a new request...")
            response = urlopen(
                request,
                *args,
                **kwargs,
            )

            logger_info("A new response is received.")
            data = load(response)
            yield GETTER.output(data)
            logger_info("New output is loaded and returned.")

    def execute(
        self,
        path: str,
        arguments: Dict[str, Json],
        inputs: Dict[str, Json],
        *args,
        urlopen=urlopen,
        **kwargs,
    ) -> HTTPResponse:

        """
        This function executes a script and returns the response.

        args and kwargs are sent to urllib.request.urlopen function.
        """

        logger_debug("Launch script execution..")
        headers = self.headers.copy()
        headers["Content-Type"] = "application/json"

        logger_debug("Build arguments...")
        arguments = {
            key: {"value": value, "input": False}
            for key, value in arguments.items()
        }
        inputs = {
            key: {"value": value, "input": True}
            for key, value in inputs.items()
        }
        arguments.update(inputs)

        logger_info("Arguments are built.")
        request = Request(
            f"{self.url}{path}",
            method="POST",
            data=dumps({"arguments": arguments}).encode(),
            headers=headers,
        )

        logger_info("Request are built. Send request...")
        return urlopen(request, *args, **kwargs)

    def upload(
        self,
        filename: str,
        file: Union[TextIOWrapper, BytesIO, bytes],
        *args,
        no_compression: bool = None,
        is_base64: bool = None,
        hidden: bool = None,
        binary: bool = None,
        read_permissions: int = None,
        write_permissions: int = None,
        delete_permissions: int = None,
        **kwargs,
    ) -> None:

        """
        This function uploads a file on WebScripts server.
        """

        logger_debug("Build headers...")
        headers = self.headers.copy()
        headers["Content-Type"] = "application/octet-stream"

        if no_compression:
            headers["No-Compression"] = "yes"

        if is_base64:
            headers["Is-Base64"] = "yes"

        if hidden:
            headers["Hidden"] = "yes"

        if binary:
            headers["Binary"] = "yes"

        if read_permissions:
            headers["Read-Permission"] = str(read_permissions)

        if write_permissions:
            headers["Write-Permission"] = str(write_permissions)

        if delete_permissions:
            headers["Delete-Permission"] = str(delete_permissions)

        if not isinstance(file, bytes):
            length = 0
            data = file.read(10240)

            while data:
                length += len(data)
                data = file.read(10240)

            file.seek(0)
            headers["Content-Length"] = str(length)

        logger_info("Headers are built. Build request...")
        request = Request(
            f"{self.url}/share/upload/{filename}",
            data=file,
            headers=headers,
            method="POST",
        )

        logger_info("Request built. Send request...")
        urlopen(
            request,
            *args,
            **kwargs,
        )
        logger_info("Get response without error. File is uploaded.")

    def download(
        self, filename: str, *args, save: bool = True, **kwargs
    ) -> GzipFile:

        """
        This function downloads a file from WebScripts server.
        """

        headers = self.headers.copy()
        headers["Accept-Encoding"] = "identity, gzip"

        logger_debug("Build download request...")
        request = Request(
            f"{self.url}/share/Download/filename/{filename}",
            headers=headers,
        )

        logger_debug("Send download request...")
        response = urlopen(
            request,
            *args,
            **kwargs,
        )

        logger_info("Get download response without error.")

        if response.headers["Content-Encoding"] == "gzip":
            logger_debug("Decompress response...")
            content = gzipopen(response)
        else:
            logger_debug("Save response in temp file...")
            content = TemporaryFile()
            copyfileobj(response, content)
            content.seek(0)

        if save:
            logger_debug("Save the contents of the downloaded file...")
            with open(filename, "wb") as output_file:
                copyfileobj(content, output_file)

        content.seek(0)
        logger_debug("Return the contents of the downloaded file...")
        return content

    def request(
        self, title: str, request: str, name: str, error_code: int = 0
    ) -> None:

        """
        This function send a request or report to the WebScripts
        Administrator.
        """

        logger_debug("Build request...")
        self.execute(
            f"/error_pages/Request/send/{error_code}",
            {
                "title": title,
                "request": request,
                "name": name,
                "error": str(error_code),
            },
            {},
        )
        logger_debug("Request sent successfully.")

    @staticmethod
    def args_command_to_webscripts(
        arguments: List[str], is_inputs: bool = False
    ) -> Dict[str, Json]:

        """
        This function returns WebScripts arguments from
        command line arguments (List[str]).
        """

        prefix = "input" if is_inputs else "arg"
        webscripts_args: Dict[str, Json] = {}
        counter = 1
        logger_debug("Parse arguments to build WebScripts arguments...")

        for argument in arguments:
            if argument.startswith("-"):
                logger_debug(f"Get optional argument: {argument}.")
                webscripts_args[argument] = True
            else:
                logger_debug(f"Get argument value: {argument}")
                webscripts_args[f"{prefix}_{counter}"] = argument
                counter += 1

        logger_info("WebScripts arguments are built.")
        return webscripts_args


def parser() -> Namespace:

    """
    This function parse command line arguments.
    """

    parser = ArgumentParser(
        description=(
            'This package is the "official" WebScripts client. This '
            "package implements client for default WebScripts features."
        )
    )
    subparsers = parser.add_subparsers(
        dest="action", help="Use default features of the WebScripts server."
    )
    add_parser = subparsers.add_parser

    information = add_parser(
        "info",
        help=(
            "Get the available scripts, arguments, and information about them."
        ),
    )
    execution = add_parser(
        "exec", help="Execute a script on WebScripts Server."
    )
    upload = add_parser("upload", help="Upload a file on WebScripts Server.")
    download = add_parser(
        "download", help="Download files from WebScripts Server."
    )
    request = add_parser(
        "request", help="Request or report to WebScripts administrator."
    )
    # test =
    add_parser("test", help="Test WebScripts server and client.")

    execution_add_argument = execution.add_argument
    execution_add_argument(
        "command", help="Command to execute script on WebScripts Server."
    )
    execution_add_argument(
        "-o",
        "--output-filename",
        "--output",
        nargs="?",
        type=FileType("w"),
        default=stdout,
        help="Output file to save the result.",
    )

    inputs = execution.add_mutually_exclusive_group()
    inputs_add_argument = inputs.add_argument
    inputs_add_argument(
        "-i",
        "--inputs",
        nargs="+",
        action="extend",
        help="Inputs value for script inputs.",
    )
    inputs_add_argument(
        "-I",
        "--input-filename",
        "--input",
        nargs="?",
        type=FileType("rb"),
        const=stdin.buffer,
        help="Input file for script inputs.",
    )

    request_add_argument = request.add_argument
    request_add_argument(
        "-s", "--subject", help="The request/report subject.", default=""
    )
    request_add_argument(
        "-n", "--name", help="Your name (Firstname LASTNAME).", default=""
    )
    request_add_argument(
        "-r", "--request", help="The request/report.", default=""
    )
    request_add_argument(
        "-c",
        "--error-code",
        help="The HTTP error code {403, 404, 406, 500 ...}.",
        type=int,
        default=0,
    )

    download_add_argument = download.add_argument
    download_add_argument(
        "-f",
        "--filenames",
        help="Filenames to download.",
        nargs="+",
        action="extend",
        required=True,
    )
    download_add_argument(
        "-o",
        "--output-filename",
        "--output",
        nargs="?",
        type=FileType("wb"),
        default=stdout.buffer,
        help="Filename to write the downloaded content.",
    )
    download_add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the download in the same local filename.",
    )

    upload_add_argument = upload.add_argument
    upload_add_argument(
        "-6",
        "--base64",
        "--64",
        action="store_true",
        help="Upload a base64 encoded file on the server.",
    )
    upload_add_argument(
        "-C",
        "--no-compression",
        action="store_true",
        help="Do not compress the file on the server.",
    )
    upload_add_argument(
        "-H",
        "--hidden",
        action="store_true",
        help="Hide the file on the server.",
    )
    upload_add_argument(
        "-b",
        "--binary",
        "--bin",
        action="store_true",
        help="Upload a binary file on the server.",
    )
    upload_add_argument(
        "-r",
        "--read-permission",
        "--read",
        type=int,
        help="Read permission on the server.",
    )
    upload_add_argument(
        "-w",
        "--write-permission",
        "--write",
        type=int,
        help="Write permission on the server.",
    )
    upload_add_argument(
        "-d",
        "--delete-permission",
        "--delete",
        type=int,
        help="Delete permission on the server.",
    )
    upload_add_argument("filename", help="The file name of the uploaded file.")

    content = upload.add_mutually_exclusive_group()
    content_add_argument = content.add_argument
    content_add_argument(
        "-f",
        "--file",
        help="The filename of the file to upload on the server.",
    )
    content_add_argument("-c", "--content", help="Content of the file.")

    information_add_argument = information.add_argument
    information_add_argument(
        "-a",
        "--no-arguments",
        action="store_true",
        help="Do not print arguments.",
    )
    information_add_argument(
        "-d",
        "--no-descriptions",
        action="store_true",
        help="Do not print description.",
    )
    information_add_argument(
        "-c",
        "--no-categories",
        action="store_true",
        help="Do not print categories.",
    )
    information_add_argument(
        "-s",
        "--scripts",
        nargs="+",
        action="extend",
        help="Information about specific scripts.",
    )

    parser_add_argument = parser.add_argument
    parser_add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode (print logs).",
    )
    parser_add_argument(
        "-i",
        "--insecure",
        action="store_true",
        help="Do not check SSL certificate.",
    )
    parser_add_argument(
        "-u",
        "--username",
        help="WebScripts username to use for this connection.",
    )
    parser_add_argument(
        "url",
        help="URL of the WebScripts server, example: http://127.0.0.1:8000",
    )

    password = parser.add_mutually_exclusive_group()
    password_add_argument = password.add_argument
    password_add_argument(
        "-p",
        "--password",
        help="WebScripts password to use for this connection.",
    )
    password_add_argument(
        "-P",
        "--password-prompt",
        action="store_true",
        help="Interactive password prompt.",
    )

    api_key = parser.add_mutually_exclusive_group()
    api_key_add_argument = api_key.add_argument
    api_key_add_argument(
        "-a",
        "--api-key",
        help="WebScripts password to use for this connection.",
    )
    api_key_add_argument(
        "-A",
        "--api-key-prompt",
        help="Interactive API key prompt.",
        action="store_true",
    )

    return parser.parse_args()


def check_connexion(function: Callable) -> Callable:

    """
    This decorator catchs urllib.error.URLError.
    """

    @wraps(function)
    def wrapper(*args, exit_on_failure: bool = False, **kwargs):

        kwds = function.__kwdefaults__
        if kwds and "exit_on_failure" in kwds:
            kwargs["exit_on_failure"] = exit_on_failure

        try:
            value = function(*args, **kwargs)
        except URLError as e:
            logger_error(
                "Error [host may be unreachable] -> "
                f"{e.__class__.__name__} {e}"
            )
            print("An URLError is raised... host may be unreachable.")

            if exit_on_failure:
                exit(4)
        else:
            return value

    return wrapper


@check_connexion
def script_execution(
    client: WebScriptsClient,
    *arguments: List[Union[str, Dict[str, Dict[str, Json]]]],
    exit_on_failure: bool = False,
) -> Iterator[Tuple[str, str, int]]:

    """
    This function calls client.execute
    and catch exceptions.
    """

    logger_debug("Launch script execution...")
    try:
        yield from client.execute_script(*arguments)
    except WebScriptsPermissionsError as e:
        logger_error(f"PermissionsError for {arguments[0]}.")
        print(
            f"Exception: {e.__class__.__name__} {e}.\n"
            "You do not have permissions, please add credentials or check it."
        )

        if exit_on_failure:
            exit(2)
    except WebScriptsScriptNotFoundError as e:
        logger_error(f"ScriptsNotFound for {arguments[0]}.")
        print(
            f"Exception: {e.__class__.__name__} {e}.\n"
            "The script used for tests is not available on this server."
        )

        if exit_on_failure:
            exit(3)


@check_connexion
def cli_execute(args: Namespace) -> None:

    """
    This function executes a script on WebScripts Server.
    """

    inputs = args.input_filename
    output = args.output_filename
    write = output.write

    logger_debug("Create WebScripts client...")
    client = WebScriptsClient(
        args.url, args.username, args.password, args.api_key
    )
    client.auth()

    logger_debug("Parse command...")
    command = list(shlex(args.command, posix=True, punctuation_chars=True))
    script_name = command.pop(0)

    inputs = (
        [line.decode("latin-1") for line in inputs.readlines()]
        if inputs
        else args.inputs
    )

    logger_debug("Get WebScriptsClient arguments and inputs from List[str]...")
    arguments = client.args_command_to_webscripts(command)

    if inputs:
        inputs = client.args_command_to_webscripts(inputs, is_inputs=True)
    else:
        inputs = {}

    logger_debug(f"Launch execution of {script_name}...")
    for o, e, c in script_execution(
        client, script_name, arguments, inputs, exit_on_failure=True
    ):
        if o.strip():
            write(f"\t{o}")

    print(f"\nExitCode: {c}\nErrors: {e}", file=output)
    logger_info("Script end.")


@check_connexion
def test(args: Namespace) -> None:

    """
    This function test WebScripts server and client.
    """

    logger_debug("Start tests...")
    script_name = "test_config.py"
    timeout = "--timeout"
    test = "test"
    password = args.password
    write = stdout.write

    arguments = {
        "select": test,
        timeout: True,
        "password": [password, password],
        "--test-date": "2016-06-22",
        "test_input": "abc",
        "test_number": 8.8,
    }
    inputs = {"test_file": "file content", "select-input": [test, "select"]}

    logger_debug("Create WebScripts client...")
    client = WebScriptsClient(args.url, args.username, password, args.api_key)
    client.auth()

    logger_info("Call API (/api/)")
    for script in client.get_scripts():
        name = script.name
        get = script.get
        print(
            f' - "{name}"\n\tCategory: "{get("category")}"'
            f'\n\tDescription: {get("description")}\n\t'
            "Arguments:\n\t\t"
            + "\n\t\t".join(
                f"{x.name.ljust(25)} (example: {x.get('example')}, type: "
                f"{x.html_type}) {x.get('description')}"
                for x in script.arguments
            )
        )

        if name == script_name:
            test_script = script

    logger_warning("Launch script test_config.py with timeout...")
    print("Execution of test_config.py with timeout.")
    for o, e, c in script_execution(
        client, script_name, arguments, inputs, exit_on_failure=True
    ):
        if o.strip():
            write(f"\t{o}")

    print(f"\nExitCode: {c}\nErrors: {e}")
    logger_info("Script end.")

    arguments.pop(timeout)

    logger_warning("Launch script test_config.py without timeout...")
    print("Execution of test_config.py without timeout.")
    for o, e, c in script_execution(
        client, test_script, arguments, inputs, exit_on_failure=True
    ):
        if o.strip():
            write(f"\t{o}")
    print(f"\nExitCode: {c}\nErrors: {e}")
    logger_info("Script end.")


@check_connexion
def cli_request(args: Namespace) -> None:

    """
    This function send request to the WebScripts Administrator.
    """

    logger_debug("Build the WebScriptsClient...")
    client = WebScriptsClient(
        args.url, args.username, args.password, args.api_key
    )
    logger_debug("Authenticate the WebScriptsClient...")
    client.auth()

    try:
        client.request(args.subject, args.request, args.name, args.error_code)
    except WebScriptsPermissionsError:
        log = (
            "Please, retry with authentication. "
            "An authentication is required on this server."
        )
        logger_error(log)
        print(log)

    logger_debug("Request sent without error.")


@check_connexion
def cli_download(args: Namespace) -> int:

    """
    This function downloads file from WebScripts server.
    """

    code = 0
    filenames = args.filenames
    output = args.output_filename
    save = args.save

    logger_debug("Build the WebScriptsClient...")
    client = WebScriptsClient(
        args.url, args.username, args.password, args.api_key
    )
    logger_debug("Authenticate the WebScriptsClient...")
    client.auth()
    download = client.download

    for filename in filenames:
        logger_debug(f"Download {filename}...")
        try:
            copyfileobj(download(filename, save=save), output)
        except WebScriptsScriptNotFoundError:
            code += 1
            logger_error(f"{filename} does not exists.")
        except WebScriptsPermissionsError:
            code += 1
            logger_error(f"You do not have permission to read on {filename}.")
        else:
            logger_info(f"{filename} is downloaded")

    return code


@check_connexion
def cli_upload(args: Namespace) -> None:

    """
    This function uploads file on WebScripts server.
    """

    code = 0
    content = args.content
    filename = args.filename
    file = args.file

    logger_debug("Get content...")
    if content:
        content = content.encode()
    elif file:
        content = open(file, "rb")
    else:
        logger_debug("No content detected, read the stdin...")
        content = BytesIO(stdin.buffer.read())

    logger_debug("Build the WebScriptsClient...")
    client = WebScriptsClient(
        args.url, args.username, args.password, args.api_key
    )
    logger_debug("Authenticate the WebScriptsClient...")
    client.auth()

    logger_debug(f"Send request to upload {filename}...")
    try:
        client.upload(
            filename,
            content,
            no_compression=args.no_compression,
            is_base64=args.base64,
            hidden=args.hidden,
            binary=args.binary,
            read_permissions=args.read_permission,
            write_permissions=args.write_permission,
            delete_permissions=args.delete_permission,
        )
    except WebScriptsPermissionsError:
        code = 1
        logger_error(f"You do not have permission to write on {filename}.")
    else:
        logger_info(f"{filename} is uploaded.")

    return code


@check_connexion
def cli_info(args: Namespace) -> None:

    """
    This function requests /api/, parse data and print it.
    """

    logger_debug("Create WebScripts client...")
    client = WebScriptsClient(
        args.url, args.username, args.password, args.api_key
    )
    client.auth()

    scripts = args.scripts
    arguments = not args.no_arguments
    categories = not args.no_categories
    description = not args.no_descriptions

    logger_info("Call API (/api/)")
    for script in client.get_scripts():
        name = script.name
        get = script.get
        logger_debug(f"Process {name}...")

        if not scripts or name in scripts:
            string = f" - {name} (Output type: {script.content_type})"

            if categories:
                string += f'\n\tCategory: "{get("category")}"'

            if description:
                string += f'\n\tDescription: {get("description")}'

            if arguments:
                string += "\n\tArguments:\n\t\t"
                string += "\n\t\t".join(
                    f'{x.name.ljust(25)} (example: "{x.get("example")}", type:'
                    f" {x.html_type}) {x.get('description')}"
                    for x in script.arguments
                )

            print(string)


def main() -> int:

    """
    This function executes this package from command line.
    """

    code = 0
    logger_debug("Start main function. Start arguments parser...")
    arguments = parser()
    logger_debug("Arguments are loaded.")

    logger.setLevel(10 if arguments.verbose else 51)
    logger_debug("Logger is configured.")

    if arguments.password_prompt:
        arguments.password = getpass("Your WebScripts password: ")

    if arguments.api_key_prompt:
        arguments.api_key = getpass("Your WebScripts API key: ")

    if arguments.insecure:
        logger_warning("Build and load an insecure SSL context...")
        global urlopen, authopener

        context = _create_unverified_context()
        urlopen = partial(urlopen, context=context)
        authopener = partial(authopener, context=context)

    action = arguments.action

    if action == "test":
        test(arguments, exit_on_failure=True)
    elif action == "info":
        cli_info(arguments, exit_on_failure=True)
    elif action == "exec":
        cli_execute(arguments, exit_on_failure=True)
    elif action == "upload":
        code = cli_upload(arguments, exit_on_failure=True)
    elif action == "download":
        code = cli_download(arguments, exit_on_failure=True)
    elif action == "request":
        cli_request(arguments, exit_on_failure=True)

    return code


if __name__ == "__main__":
    exit(main())
