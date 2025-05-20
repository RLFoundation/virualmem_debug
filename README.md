build the code and check the numa node, you can find that rocm only allocates memory perfer to node 0, but the numa node is 2. We have to wait a long time to get the memory from node 1.

```bash
numactl --hardware 
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 4
1 42 43 44 45 46 47
node 0 size: 932272 MB
node 0 free: 55862 MB
node 1 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85
86 87 88 89 90 91 92 93 94 95
node 1 size: 932335 MB
node 1 free: 730539 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
```