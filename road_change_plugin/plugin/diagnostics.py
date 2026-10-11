"""Concise diagnostic hints, without inferring task state from logs."""

def known_failure(message, detail=''):
    evidence = (str(message) + '\n' + str(detail)).lower()
    if any(s in evidence for s in ('size mismatch', 'error(s) in loading state_dict',
                                   'missing key(s) in state_dict', 'unexpected key(s) in state_dict')):
        return '模型加载失败：模型权重与当前网络结构可能不匹配。请检查配套模型文件。'
    if 'out of memory' in evidence or 'cuda error: memory' in evidence:
        return '处理所需内存或显存不足。请关闭其他占用资源的程序后重试。'
    if 'no space left' in evidence or '磁盘空间不足' in evidence:
        return '磁盘空间不足，无法写入成果。请清理成果所在磁盘后重试。'
    if 'permissionerror' in evidence or 'permission denied' in evidence:
        return '文件访问失败。请检查读写权限，以及成果文件是否被其他程序占用。'
    return ''


def failure_summary(message, detail=''):
    hint = known_failure(message, detail)
    if hint:
        return hint
    line = next((s.strip() for s in str(message).splitlines() if s.strip()), '任务未完成')
    return line[:220]

