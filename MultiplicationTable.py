def main():
    with open('result.txt', 'w') as f:
        for i in range(1, 10):
            curr = ''
            for k in range(1, i+1):
                curr = curr + str(i * k) + ' '
            curr = curr + '\n'
            f.write(curr)
        f.close()


if __name__ == '__main__':
    main()
