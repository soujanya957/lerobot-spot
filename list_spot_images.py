# list_spot_images.py

from bosdyn.client import create_standard_sdk
from bosdyn.client.image import ImageClient


def main():
    hostname = "128.148.140.22"
    username = "user"
    password = "bigbubbabigbubba"

    sdk = create_standard_sdk("spot_image_inspect")
    robot = sdk.create_robot(hostname)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = image_client.list_image_sources()

    print("Available image sources:")
    for src in sources:
        print(f"- name: {src.name}, cols: {src.cols}, rows: {src.rows}, pixel_formats: {src.pixel_formats}")


if __name__ == "__main__":
    main()




