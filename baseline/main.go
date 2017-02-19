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
	flag.IntVar(&epCount, "n", 50000, "episode count")
	flag.Parse()

	start := cuberl.RandomStates(epCount)

	var sum int
	var rewardSum int
	for _, x := range start {
		starting := x.MaxSolved
		for t := 0; t < epLen; t++ {
			x, _ = x.Move(gocube.Move(rand.Intn(18)))
		}
		sum += x.MaxSolved
		rewardSum += x.MaxSolved - starting
	}

	fmt.Println("Total solved:", float64(sum)/float64(epCount))
	fmt.Println("Reward:", float64(rewardSum)/float64(epCount))
}
