#!/usr/bin/env ghc
import Quipper

plus_minus :: Bool -> Circ Qubit
plus_minus b = do
  q <- qinit b
  r <- hadamard q
  return r

print_plus_minus :: IO ()
print_plus_minus = print_simple PDF (plus_minus False)

bell_state :: Bool -> Bool -> Circ (Qubit, Qubit)
bell_state b0 b1 = do
  q0 <- qinit b0
  q1 <- qinit b1
  hd <- hadamard q0
  bs <- qnot q1 `controlled` hd
  return (hd, bs)

main :: IO Int
main = do
  print_simple PDF (bell_state False False)
  return 0
