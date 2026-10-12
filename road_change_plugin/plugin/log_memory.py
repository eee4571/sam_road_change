"""Bounded log display; the backend log remains the full diagnostic record."""


def tail_lines(path,lines=2000,maximum_bytes=2*1024*1024):
    if not path.is_file():return []
    with path.open('rb') as stream:
        stream.seek(0,2);size=stream.tell();start=max(0,size-maximum_bytes)
        stream.seek(start);data=stream.read(maximum_bytes)
    if start:data=data.partition(b'\n')[2]
    return data.decode('utf8',errors='replace').splitlines()[-lines:]
