package handlers

import (
	"fmt"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/proxy"
)

// ProxyHandler forwards requests for voice, speaker, and health queries to the Python GPU service.
type ProxyHandler struct {
	pythonGPUURL string
}

// NewProxyHandler creates a new ProxyHandler.
func NewProxyHandler(pythonGPUURL string) *ProxyHandler {
	return &ProxyHandler{pythonGPUURL: pythonGPUURL}
}

// Forward forwards the incoming request to the target Python GPU service path.
func (p *ProxyHandler) Forward(c *fiber.Ctx) error {
	target := fmt.Sprintf("%s%s", p.pythonGPUURL, c.OriginalURL())
	if err := proxy.Do(c, target); err != nil {
		return c.Status(fiber.StatusBadGateway).JSON(fiber.Map{
			"error":  fmt.Sprintf("Failed to reach Python GPU Service: %v", err),
			"target": target,
		})
	}
	// Remove connection headers that fasthttp proxy might duplicate
	c.Response().Header.Del(fiber.HeaderServer)
	return nil
}
