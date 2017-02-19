package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/unixpickle/cuberl"
	"github.com/unixpickle/gocube"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var epLen int
	var epCount int
	flag.IntVar(&epLen, "len", 50, "episode length")
	flag.IntVar(&epCount, "n", 100, "episode count")
	flag.Parse()

	start := cuberl.RandomStates(epCount)

	var sum int
	for _, x := range start {
		for t := 0; t < epLen; t++ {
			x, _ = x.Move(gocube.Move(rand.Intn(18)))
		}
		sum += x.MaxSolved
	}

	fmt.Println(float64(sum) / float64(epCount))
}
