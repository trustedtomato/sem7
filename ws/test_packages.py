import pkg_resources

from ail_parser import Parser


def main():
    installed_packages = pkg_resources.working_set
    for package in installed_packages:
        print(package.key, package.version)


def modify_parser(parser: Parser):
    pass


if __name__ == "__main__":
    main()
