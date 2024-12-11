from src.storage_manger import ArtifactUploader


def main():
    try:
        uploader = ArtifactUploader()
        result = uploader.upload()
        print("아티팩트 업로드가 완료되었습니다.")
        return result
    except Exception as e:
        print(f"업로드 중 오류가 발생했습니다: {e}")
        return None

if __name__ == '__main__':
    main()